import random, numpy as np, pandas as pd, matplotlib.pyplot as plt, time
import torch, torch.nn as nn, torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # <-- added

# ---------- toggles ----------
USE_GRAPH_PHYSICS_TERMS = True  # set True to append [delta_size, delta_chi] to graph-level features

# ---------- (OPTIONAL) DFT baseline for speedup calc ----------
# Set this to your best estimate of the total CPU-hours your DFT workflow would take
# for an equivalent dataset/training run. If left as None, speedup won't be reported.
TOTAL_DFT_CPU_HOURS = None  # e.g., 750.0

# ---------- reproducibility ----------
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ---------- 1) load data ----------
df = pd.read_csv('vfe_dataset_with_pure.csv')
elements = ['Mo', 'Nb', 'Ta', 'V', 'W']
el2idx = {el:i for i,el in enumerate(elements)}  # canonical indices 0..4

def row_to_formula(row):
    parts = []
    for el in elements:
        f = float(row.get(f'Comp_{el}', 0.0))
        if f > 0: parts.append(f"{el}{f:.3f}")
    return "".join(parts)

df['composition_str'] = df.apply(row_to_formula, axis=1)

# ---------- 2) matminer featurization (graph-level Magpie) ----------
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element as PymatElement

# graph-level Magpie over composition
ep = ElementProperty.from_preset('magpie')
df['composition'] = df['composition_str'].apply(Composition)
df = ep.featurize_dataframe(df, col_id='composition')
magpie_cols = ep.feature_labels()

# node-level Magpie (pure element descriptors)
el_featurizer = ElementProperty.from_preset("magpie")
magpie_node_cols = el_featurizer.feature_labels()
print("Element-wise Magpie node feature length:", len(magpie_node_cols))

def idx_or_none(name):
    try: return magpie_node_cols.index(name)
    except ValueError: return None

IDX_EN   = idx_or_none('Electronegativity')
IDX_RCOV = idx_or_none('CovalentRadius')
IDX_MEND = idx_or_none('MendeleevNumber')

# ---------- 3) build node, edge, graph features ----------
node_feats_list, graph_feats_list, targets = [], [], []
edge_attr_raw_list, edge_type_id_list = [], []

# unordered chemical pair -> id (0..9) by canonical indices, fixed across all graphs
pair2id, id2pair = {}, {}
pid = 0
for i in range(5):
    for j in range(i+1, 5):
        pair2id[(i, j)] = pid
        id2pair[pid] = (i, j)
        pid += 1
NUM_EDGE_TYPES = pid  # 10
EDGE_FEAT_DIM = 6    # [frac_i*frac_j, |Δfrac|, |ΔZ|, |ΔEN|, |Δrcov|, |ΔMendeleev|]

def make_undirected_edge_index(num_nodes: int):
    """Return i<j edges once (undirected pairs)."""
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            src.append(i); dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)  # (2, nC2)

def compute_edge_features_for_nodes(nodes_np, node_canon_ids):
    """
    nodes_np[i] = [frac, Z] + magpie_feats   (no host, no pure_vfe)
    node_canon_ids[i] = canonical element index 0..4
    returns: edge_attr_undirected (nC2, EDGE_FEAT_DIM), edge_type_id (nC2,)
    """
    n = nodes_np.shape[0]

    # --- handle pure-metal (n == 1) cleanly with shape-safe arrays ---
    if n < 2:
        edge_attr_und = np.zeros((0, EDGE_FEAT_DIM), dtype=float)  
        edge_type_und = np.zeros((0,), dtype=int)                  
        return edge_attr_und, edge_type_und

    undirected = [(i, j) for i in range(n) for j in range(i+1, n)]
    feats_und, types_und = [], []

    def mag_at(idx, vec):
        if idx is None: return 0.0
        return float(vec[2 + idx])  # vec: [frac, Z] + magpie_feats

    for (i, j) in undirected:
        ni, nj = nodes_np[i], nodes_np[j]
        frac_i, Z_i = float(ni[0]), float(ni[1])
        frac_j, Z_j = float(nj[0]), float(nj[1])

        feat = [
            frac_i * frac_j,                         # composition interaction
            abs(frac_i - frac_j),                    # imbalance
            abs(Z_i - Z_j),                          # atomic number difference
            abs(mag_at(IDX_EN,   ni) - mag_at(IDX_EN,   nj)),  # EN mismatch
            abs(mag_at(IDX_RCOV, ni) - mag_at(IDX_RCOV, nj)),  # radius mismatch
            abs(mag_at(IDX_MEND, ni) - mag_at(IDX_MEND, nj)),  # Mendeleev distance
        ]
        feats_und.append(feat)

        ci, cj = node_canon_ids[i], node_canon_ids[j]
        types_und.append(pair2id[(min(ci, cj), max(ci, cj))])

    edge_attr_und = np.array(feats_und, dtype=float)  
    edge_type_und = np.array(types_und, dtype=int)     
    return edge_attr_und, edge_type_und

def physics_graph_terms(nodes_np):
    """Return [delta_size, delta_chi] as composition-weighted spreads using node Magpie (rcov, EN)."""
    def mag_at(idx, vec):
        if idx is None: return 0.0
        return float(vec[2 + idx])
    fracs = nodes_np[:,0].astype(float)
    s = fracs.sum(); fracs = fracs / (s + 1e-12)
    rcov = np.array([mag_at(IDX_RCOV, v) for v in nodes_np])
    en   = np.array([mag_at(IDX_EN,   v) for v in nodes_np])
    rbar = (fracs * rcov).sum()
    chibar = (fracs * en).sum()
    delta_size = np.sqrt(np.sum(fracs * (1.0 - rcov / (rbar + 1e-12))**2))
    delta_chi  = np.sqrt(np.sum(fracs * (en - chibar)**2))
    return np.array([delta_size, delta_chi], dtype=float)

# Build samples
num_by_nodes = {k:0 for k in range(1,6)}
for _, row in df.iterrows():
    # present elements for this composition
    present = [el for el in elements if float(row.get(f'Comp_{el}', 0.0)) > 0.0]
    if len(present) == 0:
        continue  # skip malformed rows
    num_by_nodes[len(present)] += 1

    # nodes: [frac, Z] + pure-element magpie (NO host, NO pure_VFE)
    nodes = []
    node_canon_ids = []
    for el in present:
        frac = float(row.get(f'Comp_{el}', 0.0))
        Z = float(PymatElement(el).Z)
        mag = el_featurizer.featurize(Composition(el))  # list of magpie_node_cols
        nodes.append([frac, Z] + list(map(float, mag)))
        node_canon_ids.append(el2idx[el])
    nodes_np = np.array(nodes, dtype=float)
    node_feats_list.append(nodes_np)

    # graph-level features: Magpie(comp) + Host vector (order Mo,Nb,Ta,V,W) + optional physics terms
    host_vec = np.array([float(row.get(f'Host_{el}', 0.0)) for el in elements], dtype=float)
    gfeat_base = row[magpie_cols].values.astype(float).flatten()
    if USE_GRAPH_PHYSICS_TERMS:
        gphys = physics_graph_terms(nodes_np)  # [δ, Δχ]
        gfeat = np.concatenate([gfeat_base, host_vec, gphys], axis=0)
    else:
        gfeat = np.concatenate([gfeat_base, host_vec], axis=0)
    graph_feats_list.append(gfeat)

    # target (standardize later). Pure metals are treated same as alloys (graph with one node).
    targets.append([float(row['Alloy_VFE'])])

    # undirected edges & edge types by chemical pair (shape-safe for n<2 handled inside)
    e_attr_und, e_type_und = compute_edge_features_for_nodes(nodes_np, node_canon_ids)
    edge_attr_raw_list.append(e_attr_und)   # store UNDIRECTED features (nC2, 6) or (0,6)
    edge_type_id_list.append(e_type_und)    # store UNDIRECTED types (nC2,) or (0,)

targets     = np.array(targets, dtype=float)           # (N, 1)
graph_feats = np.array(graph_feats_list, dtype=float)  # (N, 132 + 5 [+2 if physics])

# ---------- 4) standardize ----------
# Node features: variable # of nodes → concatenate all
all_nodes = np.vstack(node_feats_list)  # (sum_nodes, node_dim)
node_dim = all_nodes.shape[1]
scaler_node = StandardScaler().fit(all_nodes)
node_feats_scaled = [scaler_node.transform(n) for n in node_feats_list]

# Graph-level features
scaler_graph = StandardScaler().fit(graph_feats)
graph_feats_scaled = scaler_graph.transform(graph_feats)

# Edge features (UNDIRECTED) — safe because pure metals contribute (0,6)
if len(edge_attr_raw_list) > 0:
    all_edges = np.vstack(edge_attr_raw_list)        # (sum_undirected_edges, EDGE_FEAT_DIM)
    edge_dim = all_edges.shape[1] if all_edges.size > 0 else EDGE_FEAT_DIM
    if all_edges.size > 0:
        scaler_edge = StandardScaler().fit(all_edges)
        edge_attr_scaled_und = [scaler_edge.transform(E) if E.size > 0 else E for E in edge_attr_raw_list]
    else:
        edge_attr_scaled_und = edge_attr_raw_list
else:
    edge_dim = EDGE_FEAT_DIM
    edge_attr_scaled_und = edge_attr_raw_list

# Target
scaler_y = StandardScaler().fit(targets)
targets_std = scaler_y.transform(targets)

N = len(node_feats_scaled)
print(f"Built {N} graphs | node_dim={node_dim} | undirected edge_dim={edge_dim} | num_edge_types={NUM_EDGE_TYPES}")
print(f"Graph_feats dim: {graph_feats_scaled.shape[1]} (Magpie {len(magpie_cols)} + Host 5{' + Phys 2' if USE_GRAPH_PHYSICS_TERMS else ''})")
print("Graphs by node count:", num_by_nodes)

# ---------- 5) build PyG graphs (variable size, undirected→directed for message passing) ----------
graphs = []
for i in range(N):
    x     = torch.tensor(node_feats_scaled[i], dtype=torch.float32)      
    gfeat = torch.tensor(graph_feats_scaled[i], dtype=torch.float32)     
    y     = torch.tensor(targets_std[i],      dtype=torch.float32)      

    # UNDIRECTED pairs (i<j)
    ei_und = make_undirected_edge_index(x.size(0))                       
    if ei_und.numel() == 0:
        # single-node graph: add 1 self-loop with zero edge features/types so GAT doesn't crash
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)           
        eattr = torch.zeros((1, edge_dim), dtype=torch.float32)          
        etype = torch.tensor([0], dtype=torch.long)                      
    else:
        # Mirror undirected → both directions
        edge_index = torch.cat([ei_und, ei_und.flip(0)], dim=1)         
        # Duplicate attrs/types to match the directed edges
        eattr_und = torch.tensor(edge_attr_scaled_und[i], dtype=torch.float32)  # (nC2, edge_dim) or (0, edge_dim)
        etype_und = torch.tensor(edge_type_id_list[i],    dtype=torch.long)     # (nC2,) or (0,)
        eattr = torch.vstack([eattr_und, eattr_und]) if eattr_und.numel() > 0 else torch.zeros((0, edge_dim))
        etype = torch.hstack([etype_und, etype_und]) if etype_und.numel() > 0 else torch.zeros((0,), dtype=torch.long)

    data  = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=eattr,
        edge_type_id=etype,
        graph_feat=gfeat,
        y=y
    )
    graphs.append(data)

# ---------- 6) split & loaders ----------
train_idx, test_idx = train_test_split(range(N), test_size=0.2, random_state=SEED)
gen = torch.Generator().manual_seed(SEED)
train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=16, shuffle=True, generator=gen, num_workers=0)
test_loader  = DataLoader([graphs[i] for i in test_idx],  batch_size=16, shuffle=False, generator=gen, num_workers=0)

# ---------- 7) model (TWO GAT layers) ----------
class UltraEnhancedGAT_Physics(nn.Module):
    def __init__(self, node_in, graph_in, edge_dim, num_edge_types, hidden=256, heads=4, dropout=0.05, l1_gate=1e-4):
        super().__init__()
        self.l1_gate = l1_gate
        self.dropout = dropout

        # First GAT layer (heads=4 → hidden*heads output)
        self.gat1 = GATConv(node_in, hidden, heads=heads, concat=True,  edge_dim=edge_dim, add_self_loops=False)
        self.bn1  = nn.BatchNorm1d(hidden * heads)

        # Second GAT layer (heads=1 → hidden output)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, concat=False, edge_dim=edge_dim, add_self_loops=False)
        self.bn2  = nn.BatchNorm1d(hidden)

        # Squeeze–Excitation & residual projection
        self.se = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, hidden), nn.Sigmoid())
        self.skip_proj = nn.Linear(hidden * heads, hidden)

        # Graph-level stream
        self.lin_graph = nn.Linear(graph_in, hidden)

        # Fusion head
        self.final = nn.Sequential(
            nn.Linear(hidden*4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

        # Learnable per-element-pair gates (10 scalars: sigmoid ∈ [0,1])
        self.edge_type_logit = nn.Parameter(torch.zeros(num_edge_types))

    def _gate_edges(self, edge_attr, edge_type_id):
        gates = torch.sigmoid(self.edge_type_logit)[edge_type_id]  # (E,)
        return edge_attr * gates.unsqueeze(-1), gates

    def forward(self, data):
        x, ei, eattr, etype = data.x, data.edge_index, data.edge_attr, data.edge_type_id
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        gfeat = data.graph_feat

        eattr_s, _ = self._gate_edges(eattr, etype)

        # GAT #1
        x1 = self.gat1(x, ei, edge_attr=eattr_s)
        x1 = F.elu(self.bn1(x1))
        x1 = F.dropout(x1, self.dropout, self.training)

        # GAT #2 + residual projection to match dims
        x2 = self.gat2(x1, ei, edge_attr=eattr_s)
        x2 = F.elu(self.bn2(x2)) + self.skip_proj(x1)
        x2 = F.dropout(x2, self.dropout, self.training)

        # Squeeze–Excitation
        x2 = x2 * self.se(x2)

        # Pools
        mean_pool = global_mean_pool(x2, batch)
        max_pool  = global_max_pool(x2, batch)

        # Graph stream
        g = F.relu(self.lin_graph(gfeat.view(-1, self.lin_graph.in_features)))

        # Fusion
        h = torch.cat([mean_pool, max_pool, g, mean_pool * max_pool], dim=1)
        out = self.final(h)

        # L1 regularizer proxy on gates (mean abs of σ(logit))
        self._gate_l1 = torch.mean(torch.abs(torch.sigmoid(self.edge_type_logit)))
        return out

# ---------- 8) training (with timing) ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UltraEnhancedGAT_Physics(
    node_in=node_dim,
    graph_in=graph_feats_scaled.shape[1],
    edge_dim=edge_dim,
    num_edge_types=NUM_EDGE_TYPES,
    hidden=256, heads=4, dropout=0.05, l1_gate=1e-4
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-5, verbose=True
)

def train_epoch():
    model.train(); total = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(data).squeeze()
        loss = criterion(out, data.y.squeeze()) + model.l1_gate * model._gate_l1
        loss.backward(); optimizer.step()
        total += loss.item() * data.num_graphs
    return total / len(train_loader.dataset)

@torch.no_grad()
def eval_epoch(loader):
    model.eval(); total=0.0; preds=[]; trues=[]
    for data in loader:
        data = data.to(device)
        out = model(data).squeeze()
        loss = criterion(out, data.y.squeeze())
        total += loss.item() * data.num_graphs
        preds.append(out.detach().cpu().numpy().reshape(-1,1))
        trues.append(data.y.detach().cpu().numpy().reshape(-1,1))
    return total/len(loader.dataset), np.vstack(preds), np.vstack(trues)

best_val = float('inf'); best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
wait, PATIENCE = 0, 60

epoch_times = []
train_start = time.perf_counter()

for epoch in range(1, 301):
    t0 = time.perf_counter()
    tr = train_epoch()
    va,_,_ = eval_epoch(test_loader)
    scheduler.step(va)
    epoch_times.append(time.perf_counter() - t0)

    improved = va < best_val - 1e-6
    if improved:
        best_val = va; wait = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train {tr:.6f} | Val {va:.6f} | LR {optimizer.param_groups[0]['lr']:.2e} | epoch {epoch_times[-1]:.2f}s")
    if wait >= PATIENCE:
        print("Early stopping."); break

total_train_s = time.perf_counter() - train_start
avg_epoch_s = np.mean(epoch_times) if epoch_times else float('nan')
print(f"\nTraining time: {total_train_s:.2f}s total | {avg_epoch_s:.2f}s/epoch (avg over {len(epoch_times)} epochs)")

# load best
model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# ---------- 9) final evaluation + inference timing ----------
@torch.no_grad()
def timed_inference(loader):
    model.eval()
    # warm-up (especially for GPU)
    for _ in range(2):
        for data in loader:
            data = data.to(device)
            _ = model(data)
        break
    # timed pass
    n_graphs = 0
    t0 = time.perf_counter()
    for data in loader:
        n_graphs += data.num_graphs
        data = data.to(device)
        _ = model(data)
    dt = time.perf_counter() - t0
    return dt, n_graphs

val_loss, preds, trues = eval_epoch(test_loader)
infer_s, n_eval = timed_inference(test_loader)

pred_phys  = scaler_y.inverse_transform(preds)
trues_phys = scaler_y.inverse_transform(trues)

rmse = np.sqrt(mean_squared_error(trues_phys.flatten(), pred_phys.flatten()))
mae  = mean_absolute_error(trues_phys.flatten(), pred_phys.flatten())
r2   = r2_score(trues_phys.flatten(), pred_phys.flatten())

throughput = n_eval / infer_s if infer_s > 0 else float('inf')
per_graph_ms = 1000.0 * infer_s / n_eval if n_eval > 0 else float('nan')

print(f"\nTest RMSE: {rmse:.6f} eV | Test MAE: {mae:.6f} eV | Test R²: {r2:.6f}")
print(f"Inference: {infer_s:.3f}s over {n_eval} graphs | ~{throughput:.1f} graphs/s | ~{per_graph_ms:.2f} ms/graph")

if TOTAL_DFT_CPU_HOURS is not None:
    dft_seconds = TOTAL_DFT_CPU_HOURS * 3600.0
    speedup_train_only = dft_seconds / total_train_s if total_train_s > 0 else float('inf')
    speedup_train_plus_eval = dft_seconds / (total_train_s + infer_s) if (total_train_s + infer_s) > 0 else float('inf')
    print(f"\nEstimated speedup vs. DFT baseline:")
    print(f"  vs. training only:      ~{speedup_train_only:,.0f}×")
    print(f"  vs. train + evaluation: ~{speedup_train_plus_eval:,.0f}×")
    print("  (Set TOTAL_DFT_CPU_HOURS to your workflow’s CPU-hour estimate.)")

plt.figure(figsize=(6,6))
plt.scatter(trues_phys.flatten(), pred_phys.flatten(), alpha=0.6, edgecolor='k', linewidth=0.4)
mn, mx = trues_phys.min(), trues_phys.max()
plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.2)
plt.xlabel('Actual VFE (eV)'); plt.ylabel('Predicted VFE (eV)')
plt.title('UltraEnhancedGAT (2-layer, undirected pairs, host@graph, no pure-VFE input)')
plt.grid(True, ls='--', alpha=0.5); plt.tight_layout(); plt.show()
