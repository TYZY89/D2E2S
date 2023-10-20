import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GCN2Conv, TAGConv, ChebConv, GatedGraphConv, ResGatedGraphConv
from torch_geometric.utils import dense_to_sparse

class TIN(nn.Module):
    def __init__(self, hidden_dim):
        super(TIN, self).__init__()

        self.hidden_dim = hidden_dim

        # Define residual connections and LayerNorm layers
        self.residual_layer1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))
        self.residual_layer4 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))

        self.GatedGCN = GatedGCN(hidden_dim, hidden_dim)

        # Fusion layer
        self.lstm = nn.LSTM(self.hidden_dim*2, self.hidden_dim, 2, batch_first=True,
                            bidirectional=True)

        # MLP
        self.feature_fusion = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim),
                                             nn.LayerNorm(hidden_dim))

    def forward(self, h_feature, h_syn_ori, h_syn_feature, h_sem_ori, h_sem_feature, adj_sem_ori, adj_sem_gcn):

        # (batch_size, sequence_length, hidden_dim)
        assert h_feature.shape == h_syn_feature.shape == h_sem_ori.shape == h_sem_feature.shape
        assert len(h_feature.shape) == 3

        # residual layer
        h_syn_origin = self.residual_layer1(h_feature + h_syn_ori)
        h_syn_feature = self.residual_layer2(h_feature + h_syn_feature)
        h_sem_origin = self.residual_layer3(h_feature + h_sem_ori)
        h_sem_feature = self.residual_layer4(h_feature + h_sem_feature)

        # h_syn_origin, h_syn_feature = self.GatedGCN(h_syn_origin, h_syn_feature, adj_sem_ori, adj_sem_gcn)
        # h_sem_origin, h_sem_feature = self.GatedGCN(h_sem_origin, h_sem_feature, adj_sem_ori, adj_sem_gcn)

        concat = torch.cat([h_syn_feature, h_sem_feature], dim=2)
        output, _ = self.lstm(concat)
        h_fusion = self.feature_fusion(output)

        return h_fusion


class FeatureStacking(nn.Module):
    def __init__(self, hidden_dim):
        super(FeatureStacking, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, input1, input2):
        # stack the three input features along the third dimension to form a new tensor with dimensions [a, b, c, hidden_dim]
        # stacked_input = torch.stack([input1, input2, input3, input4], dim=3)
        stacked_input = torch.stack([input1, input2], dim=3)

        # apply average pooling along the fourth dimension to obtain a tensor with dimensions [a, b, c, 1]
        # pooled_input = torch.mean(stacked_input, dim=3, keepdim=True)
        pooled_input,_ = torch.max(stacked_input, dim=3, keepdim=True)

        # reshape the tensor to the desired output shape [a, b, c]
        output = pooled_input.squeeze(3)

        return output

class GatedGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, gated_layers=2):
        super(GatedGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gated_layers = gated_layers
        self.conv1 = GCNConv(self.input_dim, self.hidden_dim)                                                           # GCNConv默认添加add_self_loops
        self.conv2 = GCNConv(self.input_dim, self.hidden_dim)
        self.conv3 = GatedGraphConv(self.hidden_dim, self.gated_layers)

    def forward(self, input1, input2, adj_sem_ori, adj_sem_gcn):
        # Build graph data structures
        input1_ = input1.view(-1)
        input2_ = input2.view(-1)
        features = torch.stack([input1_, input2_], dim=0)
        data = Data(x=features)
        data.cuda()
        data.x = data.x.view(-1, self.input_dim)
        data.edge_index, _ = dense_to_sparse(torch.ones((input1.size(1), input1.size(1))).cuda())
        data.edge_attr = compute_cosine_similarity(data.x, data.edge_index)
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = Multi_Head_S_Pool(x, adj_sem_ori, adj_sem_gcn)
        # x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index))
        h_fusion_1, h_fusion_2 = x.view(2, 16, -1, 768)[0], x.view(2, 16, -1, 768)[1]
        return h_fusion_1, h_fusion_2

def Multi_Head_S_Pool(x, adj_sem_ori, adj_sem_gcn):

    # Average pooling
    adj = torch.cat([adj_sem_ori, adj_sem_gcn], dim=0)
    S_Mean = adj.mean(dim=2)  # Compute the mean along the seq_len dimension
    S_Mean = S_Mean.view(x.size(0), -1)  # Reshape to match the dimensions of x

    # Max pooling
    S_Max = adj.max(dim=2).values  # Take the max along the seq_len dimension
    S_Max = S_Max.view(x.size(0), -1)  # Reshape to match the dimensions of x

    # Calculate Z_1
    Z_1 = F.relu(x * (1 + S_Mean + S_Max))

    return Z_1

# Calculate the edge weights, i.e. Euclidean distance
def edge_weight(x, edge_index):
    row, col = edge_index
    edge_attr = (x[row] - x[col]).norm(p=2, dim=-1).view(edge_index.size(1), -1)

    return edge_attr

# Cosine similarity
def compute_cosine_similarity(x, edge_index):

    edge_index_row, edge_index_col = edge_index[0], edge_index[1]

    x_row = x[edge_index_row]
    x_col = x[edge_index_col]
    similarity = F.cosine_similarity(x_row, x_col, dim=1)
    min_value = similarity.min()
    max_value = similarity.max()
    similarity = (similarity - min_value) / (max_value - min_value)

    return similarity

# Pearson correlation coefficient
def compute_pearson_correlation(x, edge_index):
    mean_x = torch.mean(x, dim=1)

    # Compute differences between each value and the mean for x
    diff_x = x - mean_x[:, None]

    # Compute the sum of squared differences for x
    sum_squared_diff_x = torch.sum(diff_x ** 2, dim=1)

    # Compute the square root of the sum of squared differences for x
    sqrt_sum_squared_diff_x = torch.sqrt(sum_squared_diff_x)

    # Compute the product of the square roots for x
    product_sqrt_diff_x = sqrt_sum_squared_diff_x[edge_index[0]] * sqrt_sum_squared_diff_x[edge_index[1]]

    # Compute the sum of the multiplied differences
    sum_multiplied_diff = torch.sum(diff_x[edge_index[0]] * diff_x[edge_index[1]], dim=1)

    # Compute the Pearson correlation coefficient
    pearson_corr = sum_multiplied_diff / product_sqrt_diff_x

    return pearson_corr






