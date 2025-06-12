import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('vfe_dataset_with_pure.csv')

comp_cols = [c for c in df.columns if c.startswith('Comp_')]
host_cols = [c for c in df.columns if c.startswith('Host_')]
feature_cols = comp_cols + ['Pure_VFE'] + host_cols

X = df[feature_cols].values
y = df['Alloy_VFE'].values.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

def create_graph_dataset(X, y):
    dataset = []
    for i in range(len(X)):
        x = torch.tensor(X[i], dtype=torch.float).unsqueeze(0)  
        edge_index = torch.empty((2, 0), dtype=torch.long)      
        y_tensor = torch.tensor(y[i], dtype=torch.float)
        dataset.append(Data(x=x, edge_index=edge_index, y=y_tensor))
    return dataset

train_dataset = create_graph_dataset(X_train, y_train)
test_dataset = create_graph_dataset(X_test, y_test)

from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

class SimpleGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return x.squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

input_dim = X_train.shape[1]
model = SimpleGNN(input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = nn.MSELoss()

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y.squeeze())
            total_loss += loss.item() * batch.num_graphs
            preds.append(out.cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader.dataset), preds, trues

num_epochs = 200
for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, preds, trues = evaluate(model, test_loader, criterion)
    if epoch % 10 == 0 or epoch == 1:
        r2 = r2_score(scaler_y.inverse_transform(trues.reshape(-1, 1)),
                      scaler_y.inverse_transform(preds.reshape(-1, 1)))
        print(f"Epoch {epoch:03d}: Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val R² {r2:.4f}")

val_loss, preds, trues = evaluate(model, test_loader, criterion)
preds_orig = scaler_y.inverse_transform(preds.reshape(-1, 1))
trues_orig = scaler_y.inverse_transform(trues.reshape(-1, 1))

print(f"\nFinal Test RMSE: {np.sqrt(val_loss):.4f} eV")
print(f"Final Test R²: {r2_score(trues_orig, preds_orig):.4f}")

results_df = pd.DataFrame({
    'Actual_VFE': trues_orig.flatten(),
    'Predicted_VFE': preds_orig.flatten(),
    'Percent_Diff': 100 * np.abs((trues_orig.flatten() - preds_orig.flatten()) / trues_orig.flatten())
})
results_df.to_csv('gnn_vfe_predictions.csv', index=False)
print("Saved predictions to 'gnn_vfe_predictions.csv'.")

plt.figure(figsize=(6,6))
plt.scatter(trues_orig, preds_orig, alpha=0.6, edgecolor='k')
plt.plot([trues_orig.min(), trues_orig.max()], [trues_orig.min(), trues_orig.max()], 'r--')
plt.xlabel('Actual Vacancy Formation Energy (eV)')
plt.ylabel('Predicted Vacancy Formation Energy (eV)')
plt.title('GNN: Actual vs Predicted VFE')
plt.grid(True)
plt.tight_layout()
plt.show()
