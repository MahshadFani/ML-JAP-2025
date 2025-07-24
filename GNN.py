import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from matminer.featurizers.composition import ElementProperty
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element as PymatElement
import matplotlib.pyplot as plt

# 1. Load data and build alloy formula for matminer
df = pd.read_csv('vfe_dataset_with_pure.csv')
elements = ['Mo', 'Nb', 'Ta', 'V', 'W']
comp_cols = [f'Comp_{el}' for el in elements]

def row_to_formula(row):
    return "".join(f"{el}{row[f'Comp_{el}']:.3f}" for el in elements if row[f'Comp_{el}'] > 0)

df['composition_str'] = df.apply(row_to_formula, axis=1)

# 2. Matminer featurization (graph-level)
ep = ElementProperty.from_preset('magpie')
df['composition'] = df['composition_str'].apply(Composition)
df = ep.featurize_dataframe(df, col_id='composition')
magpie_cols = ep.feature_labels()

# 2b. Prepare per-element Magpie features for node features
el_featurizer = ElementProperty.from_preset("magpie")
magpie_node_cols = el_featurizer.feature_labels()
print("Element-wise Magpie node feature length:", len(magpie_node_cols))

# 3. Prepare node and graph features (with per-node Magpie)
node_feats_list = []
graph_feats_list = []
targets = []

for i, row in df.iterrows():
    nodes = []
    for el in elements:
        atom_num = PymatElement(el).Z
        magpie_feats = el_featurizer.featurize(Composition(el))
        nodes.append([row[f'Comp_{el}'], row['Pure_VFE'], row[f'Host_{el}'], atom_num] + list(magpie_feats))
    node_feats_list.append(nodes)
    graph_feats_list.append(row[magpie_cols].values.astype(float).flatten())
    targets.append([row['Alloy_VFE']])

node_feats  = np.array(node_feats_list, dtype=float)   # (N,5,4+len(magpie_node_cols))
graph_feats = np.array(graph_feats_list, dtype=float)  # (N,num_magpie)
targets     = np.array(targets, dtype=float)           # (N,1)

print("Node feature shape:", node_feats.shape)
print("Graph_feats shape:", graph_feats.shape)

# 4. Standardize features
N, num_nodes, node_dim = node_feats.shape
node_flat = node_feats.reshape(-1, node_dim)
node_flat = StandardScaler().fit_transform(node_flat)
node_feats = node_flat.reshape(N, num_nodes, node_dim)
graph_feats = StandardScaler().fit_transform(graph_feats)
scaler_y    = StandardScaler()
targets_std = scaler_y.fit_transform(targets)

# 5. Build Data objects for PyG
idx = torch.combinations(torch.arange(num_nodes), r=2).t()
edge_index = torch.cat([idx, idx.flip(0)], dim=1)

graphs = []
for i in range(N):
    x     = torch.tensor(node_feats[i],      dtype=torch.float)
    gfeat = torch.tensor(graph_feats[i],     dtype=torch.float)
    y     = torch.tensor(targets_std[i],     dtype=torch.float)
    graphs.append(Data(x=x, edge_index=edge_index,
                       graph_feat=gfeat, y=y))

print("Node feature shape for first graph:", node_feats[0].shape)
print("Graph-level feature shape for first graph:", graph_feats[0].shape)
print("Parameter for graph_in:", graph_feats.shape[1])

# 6. Train/test split and loaders
train_idx, test_idx = train_test_split(range(N), test_size=0.2, random_state=42)
train_loader = DataLoader([graphs[i] for i in train_idx], batch_size=16, shuffle=True)
test_loader  = DataLoader([graphs[i] for i in test_idx],  batch_size=16, shuffle=False)

# 7. UltraEnhancedGCN model (GNN version)
class UltraEnhancedGCN(nn.Module):
    def __init__(self, node_in, graph_in, hidden=256, dropout=0.1):
        super().__init__()
        self.gcn1 = GCNConv(node_in, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.gcn3 = GCNConv(hidden, hidden)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.gcn4 = GCNConv(hidden, hidden)
        self.bn4 = nn.BatchNorm1d(hidden)
        self.lin_graph = nn.Linear(graph_in, hidden)
        self.final = nn.Sequential(
            nn.Linear(hidden*4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.dropout = dropout
        self.se = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden),
            nn.Sigmoid()
        )
        self.skip_proj = nn.Linear(hidden, hidden)

    def forward(self, data):
        x, edge_index, batch, gfeat = (
            data.x, data.edge_index, data.batch, data.graph_feat
        )
        x1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.bn2(self.gcn2(x1, edge_index))) + x1
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = F.relu(self.bn3(self.gcn3(x2, edge_index))) + x2
        x3 = F.dropout(x3, self.dropout, training=self.training)
        x3_proj = self.skip_proj(x3)
        x4 = F.relu(self.bn4(self.gcn4(x3, edge_index))) + x3_proj
        x4 = F.dropout(x4, self.dropout, training=self.training)
        attn = self.se(x4)
        x4 = x4 * attn
        mean_pool = global_mean_pool(x4, batch)
        max_pool = global_max_pool(x4, batch)
        expected_features = self.lin_graph.in_features
        if gfeat.dim() == 1:
            if gfeat.shape[0] == expected_features * mean_pool.shape[0]:
                gfeat = gfeat.view(mean_pool.shape[0], expected_features)
            else:
                gfeat = gfeat.unsqueeze(0)
        elif gfeat.dim() == 2 and gfeat.shape[0] != mean_pool.shape[0]:
            gfeat = gfeat.reshape(mean_pool.shape[0], expected_features)
        g = F.relu(self.lin_graph(gfeat))
        h = torch.cat([mean_pool, max_pool, g, mean_pool * max_pool], dim=1)
        return self.final(h)

# 8. Training and evaluation functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UltraEnhancedGCN(node_in=node_feats.shape[2], graph_in=graph_feats.shape[1], hidden=256, dropout=0.1).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

def train_epoch():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def eval_epoch():
    model.eval()
    total_loss, preds, trues = 0, [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            total_loss += criterion(out.squeeze(), data.y).item() * data.num_graphs
            preds.append(out.cpu().numpy())
            trues.append(data.y.cpu().numpy())
    return (total_loss / len(test_loader.dataset),
            np.vstack(preds), np.vstack(trues))

# 9. Training loop
for epoch in range(1, 201):
    tr_loss = train_epoch()
    val_loss, preds, trues = eval_epoch()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Train Loss {tr_loss:.4f} | Val Loss {val_loss:.4f}")

# 10. Final evaluation and metrics
val_loss, preds, trues = eval_epoch()
pred_phys = scaler_y.inverse_transform(preds)
trues_phys = scaler_y.inverse_transform(trues)

rmse = np.sqrt(mean_squared_error(trues_phys.flatten(), pred_phys.flatten()))
r2   = r2_score(trues_phys.flatten(), pred_phys.flatten())
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R²: {r2:.4f}")

plt.figure(figsize=(6,6))
plt.scatter(trues_phys.flatten(), pred_phys.flatten(), alpha=0.6, edgecolor='k')
plt.plot([trues_phys.min(), trues_phys.max()],
         [trues_phys.min(), trues_phys.max()], 'r--')
plt.xlabel('Actual VFE (eV)')
plt.ylabel('Predicted VFE (eV)')
plt.title('UltraEnhancedGCN: Actual vs Predicted VFE')
plt.grid(True)
plt.tight_layout()
plt.show()

