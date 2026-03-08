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
        
        # Multi-Head Self Attention mechanism across the 5 streams
        self.embedding_dim = hidden_dim // 2
        # Project each to a sequence of 5 tokens
        self.mha = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Final weight projector
        self.weight_projector = nn.Sequential(
            nn.Linear(self.embedding_dim * 5, 5),
            nn.Softmax(dim=1)
        )
        
    def forward(self, clinical_data, us_data, genomic_data, path_data, sensor_data):
        c_feat = self.clinical_encoder(clinical_data)
        u_feat = self.us_encoder(us_data)
        g_feat = self.genomic_encoder(genomic_data)
        p_feat = self.path_encoder(path_data)
        s_feat = self.sensor_encoder(sensor_data)
        
        # Stack into sequence for Transformer: [Batch, Sequence=5, Embedding]
        seq = torch.stack([c_feat, u_feat, g_feat, p_feat, s_feat], dim=1)
        
        # Apply Multi-Head Attention
        attn_out, _ = self.mha(seq, seq, seq)
        # Add & Norm
        seq = self.layer_norm(seq + attn_out)
        
        # Flatten and project to attention weights
        flat_seq = seq.view(seq.shape[0], -1)
        weights = self.weight_projector(flat_seq)
        
        return c_feat, u_feat, g_feat, p_feat, s_feat, weights

if __name__ == "__main__":
    model = FeatureWeightingFFNN()
    # Dummy data test (all 5 modalities)
    c_dummy = torch.randn(10, 9)
    u_dummy = torch.randn(10, 128)
    g_dummy = torch.randn(10, 256)
    p_dummy = torch.randn(10, 64)
    s_dummy = torch.randn(10, 32)
    c_out, u_out, g_out, p_out, s_out, w_out = model(c_dummy, u_dummy, g_dummy, p_dummy, s_dummy)
    print("Clinical features:", c_out.shape)
    print("US features:", u_out.shape)
    print("Modality weights:", w_out.shape)
