import torch
import os
import glob
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import PI_STGNN, balanced_physics_informed_loss

# --- 1. Dataset Initialization ---
class HistoricalClimateDataset(Dataset):
    def __init__(self, tensor_dir, seq_length=5):
        self.seq_length = seq_length
        x_files = sorted(glob.glob(os.path.join(tensor_dir, 'X_*.pt')))
        y_files = sorted(glob.glob(os.path.join(tensor_dir, 'Y_*.pt')))
        self.X = torch.cat([torch.load(f) for f in x_files], dim=0)
        self.Y = torch.cat([torch.load(f) for f in y_files], dim=0)
        
    def __len__(self): return len(self.X) - self.seq_length
    def __getitem__(self, idx): return self.X[idx : idx + self.seq_length], self.Y[idx + self.seq_length]

if __name__ == "__main__":
    tensor_dir = "../data/processed_tensors/"
    full_dataset = HistoricalClimateDataset(tensor_dir=tensor_dir)
    
    train_size = int(len(full_dataset) * 0.8)
    test_size = len(full_dataset) - train_size
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
    test_dataset = torch.utils.data.Subset(full_dataset, range(train_size, train_size + test_size))

    # --- 2. Normalization ---
    train_X_raw = full_dataset.X[train_dataset.indices]
    feature_means = train_X_raw.mean(dim=(0, 1), keepdim=True)
    feature_stds = train_X_raw.std(dim=(0, 1), keepdim=True)
    full_dataset.X = (full_dataset.X - feature_means) / (feature_stds + 1e-7)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Calculate Weights
    num_extreme = (full_dataset.Y[train_dataset.indices] == 1).sum().item()
    num_normal = (full_dataset.Y[train_dataset.indices] == 0).sum().item()
    dynamic_pos_weight = torch.tensor((num_normal / num_extreme) * 0.5) 

    # --- 3. Training Loop ---
    model = PI_STGNN(num_node_features=4, hidden_dim=64, seq_length=5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    rows = torch.arange(11).repeat_interleave(11)
    cols = torch.arange(11).repeat(11)
    edge_index = torch.stack([rows, cols], dim=0)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x, edge_index)
            
            unscaled_humidity = (batch_x[:, -1, :, 1] * feature_stds[0, 0, 1]) + feature_means[0, 0, 1]
            unscaled_z500 = (batch_x[:, -1, :, 0] * feature_stds[0, 0, 0]) + feature_means[0, 0, 0]
            
            loss = balanced_physics_informed_loss(
                predictions, batch_y, unscaled_humidity.mean(dim=1), unscaled_z500.mean(dim=1) * 0.1, 0.5, dynamic_pos_weight
            )
            loss.backward()
            optimizer.step()

    os.makedirs("../outputs", exist_ok=True)
    torch.save(model.state_dict(), '../outputs/PI_STGNN_Midwest_40Yr.pth')
    print("Training complete. Model saved.")
