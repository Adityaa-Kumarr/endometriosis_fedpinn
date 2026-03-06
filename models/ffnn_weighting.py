import torch
import torch.nn as nn

class FeatureWeightingFFNN(nn.Module):
    """
    Feed-Forward Neural Network to compute feature importance weights 
    from multi-modal data (clinical, hormonal, ultrasound embeddings).
    """
    def __init__(self, clinical_dim=9, us_dim=128, genomic_dim=256, path_dim=64, sensor_dim=32, hidden_dim=64):
        super(FeatureWeightingFFNN, self).__init__()
        
        # Encoders for each multi-modal data stream
        self.clinical_encoder = nn.Sequential(nn.Linear(clinical_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2))
        self.us_encoder = nn.Sequential(nn.Linear(us_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2))
        self.genomic_encoder = nn.Sequential(nn.Linear(genomic_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2))
        self.path_encoder = nn.Sequential(nn.Linear(path_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2))
        self.sensor_encoder = nn.Sequential(nn.Linear(sensor_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, hidden_dim // 2))
        
        # Attention/Weighting mechanism across all 5 streams
        input_dim_for_attention = (hidden_dim // 2) * 5
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim_for_attention, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # 5 modalities: clinical, US, genomic, pathology, sensor
            nn.Softmax(dim=1) 
        )
        
    def forward(self, clinical_data, us_data, genomic_data, path_data, sensor_data):
        c_feat = self.clinical_encoder(clinical_data)
        u_feat = self.us_encoder(us_data)
        g_feat = self.genomic_encoder(genomic_data)
        p_feat = self.path_encoder(path_data)
        s_feat = self.sensor_encoder(sensor_data)
        
        combined_features = torch.cat([c_feat, u_feat, g_feat, p_feat, s_feat], dim=1)
        
        # Compute dynamic weights for each modality
        weights = self.attention_weights(combined_features)
        
        return c_feat, u_feat, g_feat, p_feat, s_feat, weights

if __name__ == "__main__":
    model = FeatureWeightingFFNN()
    # Dummy data test
    c_dummy = torch.randn(10, 9)
    u_dummy = torch.randn(10, 128)
    c_out, u_out, w_out = model(c_dummy, u_dummy)
    print("Clinical features:", c_out.shape)
    print("US features:", u_out.shape)
    print("Modality weights:", w_out.shape)
