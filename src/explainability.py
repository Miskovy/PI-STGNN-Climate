import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import PI_STGNN

if __name__ == "__main__":
    model = PI_STGNN(num_node_features=4, hidden_dim=64, seq_length=5)
    model.load_state_dict(torch.load('../outputs/PI_STGNN_Midwest_40Yr.pth'))
    model.eval()

    rows = torch.arange(11).repeat_interleave(11)
    cols = torch.arange(11).repeat(11)
    edge_index = torch.stack([rows, cols], dim=0)

    # Synthetic Storm Generation for Attribution
    target_sequence = torch.zeros((1, 5, 11, 4))
    target_sequence[0, 0, :, 0] = 2.5 
    target_sequence[0, 1, :, 0] = 1.8 
    target_sequence[0, 3, :, 1] = 1.5 
    target_sequence[0, 4, :, 1] = 3.0 

    target_sequence = target_sequence.clone().detach().requires_grad_(True)
    prediction = model(target_sequence, edge_index)
    prediction.backward()

    gradients = target_sequence.grad.abs()
    regional_importance = gradients[0].mean(dim=1).numpy()
    regional_importance = regional_importance / (np.sum(regional_importance) + 1e-8)

    features = ["Z500", "Humidity", "U-Wind", "V-Wind"]
    days = ["Day t-4", "Day t-3", "Day t-2", "Day t-1", "Day t-0"]

    plt.figure(figsize=(10, 6))
    sns.heatmap(regional_importance.T, annot=True, cmap="YlOrRd", fmt=".3f", 
                xticklabels=days, yticklabels=features)
    
    plt.title("Spatiotemporal Feature Attribution")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("../outputs/Feature_Attribution_Heatmap.png", dpi=300)
    print("Heatmap saved to outputs/")
