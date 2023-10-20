import torch
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.nn.functional as F

def Orthographic_projection_fusion(feature1, feature2, feature3):
    # Dimensionality reduction is performed on each feature to obtain a low-dimensional feature vector
    pca = PCA(n_components=768)
    feature1_vector = pca.fit_transform(feature1.view(-1, 768)).reshape(4, 24, -1)
    feature2_vector = pca.fit_transform(feature2.view(-1, 768)).reshape(4, 24, -1)
    feature3_vector = pca.fit_transform(feature3.view(-1, 768)).reshape(4, 24, -1)

    # The vectors of all features are combined to obtain a combined feature vector
    fused_feature_vector = torch.cat([feature1_vector, feature2_vector, feature3_vector], dim=-1)

    # Projection of the integrated feature vector into a new space using the projection matrix selected by PCA
    projection_matrix = torch.from_numpy(pca.components_).to(torch.float32)
    fused_feature_vector = torch.matmul(fused_feature_vector.view(-1, 2304), projection_matrix.T).view(4, 24, -1)

    # Return the fused feature vector
    return fused_feature_vector

class TextCentredSP(nn.Module):
    def __init__(self, input_dims, shared_dims, private_dims):
        super(TextCentredSP, self).__init__()
        self.input_dims = input_dims
        self.shared_dims = shared_dims
        self.private_dims = private_dims

        # Shared Semantic Mask Matrix
        self.shared_mask = nn.Parameter(torch.ones(self.input_dims))
        # Personalized Semantic Mask Matrix
        self.private_mask = nn.Parameter(torch.ones(self.input_dims))

        # Shared Semantic Encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.shared_dims),
            nn.ReLU()
        )

        # Personalized Semantic Encoder
        self.private_encoder = nn.Sequential(
            nn.Linear(self.input_dims, self.private_dims),
            nn.ReLU()
        )

    def forward(self, h_syn_ori, h_syn_feature):
        # Stitching together the features of the three modalities
        features = torch.cat((h_syn_ori, h_syn_feature), dim=2)

        # Calculating the shared semantic mask matrix
        shared_weights = F.softmax(self.shared_mask.view(-1), dim=0).view(self.input_dims)
        shared_mask = shared_weights > 0.2  # threshold
        shared_mask = shared_mask.float()

        # Calculate the personality semantic mask matrix
        private_mask = 1 - shared_mask

        # Masking of the features of the three modalities
        shared_features = features * shared_mask
        private_features = features * private_mask

        # Encoding shared semantic and individual semantic features
        shared_code = self.shared_encoder(shared_features)
        private_code = self.private_encoder(private_features)

        # Shared semantic and individual semantic features after merged encoding
        output = torch.cat((shared_code, private_code), dim=2)

        return output