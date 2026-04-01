import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class PI_STGNN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, seq_length):
        super(PI_STGNN, self).__init__()
        self.seq_length = seq_length
        self.gcn = GCNConv(num_node_features, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, edge_index):
        batch_size = x_seq.shape[0]
        batch_predictions = []
        for b in range(batch_size):
            lstm_inputs = []
            for t in range(self.seq_length):
                x_t = x_seq[b, t, :, :] 
                out_t = F.relu(self.gcn(x_t, edge_index))
                graph_embedding = torch.mean(out_t, dim=0) 
                lstm_inputs.append(graph_embedding)
                
            lstm_inputs = torch.stack(lstm_inputs) 
            lstm_out, (hn, cn) = self.lstm(lstm_inputs.unsqueeze(0))
            
            final_state = hn[-1] 
            prediction = torch.sigmoid(self.classifier(final_state))
            batch_predictions.append(prediction)
            
        return torch.cat(batch_predictions, dim=0)

def balanced_physics_informed_loss(prediction, targets, humidity_data, temp_data, lambda_pi=0.1, pos_weight=1.0):
    bce_loss = - (pos_weight * targets * torch.log(prediction + 1e-7) + 
                  (1 - targets) * torch.log(1 - prediction + 1e-7)).mean()
    
    theoretical_max_humidity = temp_data * 0.07 
    physics_violation = torch.relu(humidity_data - theoretical_max_humidity)
    physics_penalty = torch.mean(physics_violation)
    
    return bce_loss + (lambda_pi * physics_penalty)
