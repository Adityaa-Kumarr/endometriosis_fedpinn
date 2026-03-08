import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.block(x)) # Skip connection

class MoELayer(nn.Module):
    """
    Mixture of Experts layer dynamically routing inputs to top-k specialized
    expert networks based on a gating mechanism.
    """
    def __init__(self, hidden_dim, num_experts=4, k=2):
        super(MoELayer, self).__init__()
        self.k = k
        self.num_experts = num_experts
        # Experts: e.g. Ovarian, DIE, Superficial, Adenomyosis specialized blocks
        self.experts = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_experts)])
        # Gating network
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Calculate gating probabilities
        gate_logits = self.gate(x)
        gate_probs = self.softmax(gate_logits)
        
        # Get top-k experts and their weights (prob)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=1)
        
        # Normalize top-k probabilities so they sum to 1
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=1, keepdim=True) + 1e-9)
        
        # Output tensor initialized to 0
        moe_out = torch.zeros_like(x)
        
        # Loop over experts and sum weighted outputs
        for i in range(self.num_experts):
            # Mask to see if expert 'i' is in the top-k for each item in batch
            expert_mask = (top_k_indices == i)
            # Find the position (0 or 1) in top-k where this expert was selected
            batch_indices, k_indices = torch.nonzero(expert_mask, as_tuple=True)
            
            if batch_indices.numel() > 0:
                expert_inputs = x[batch_indices]
                expert_outputs = self.experts[i](expert_inputs)
                # Weight by the normalized gate probabilities
                expert_weights = top_k_probs[batch_indices, k_indices].unsqueeze(1)
                moe_out[batch_indices] += expert_outputs * expert_weights
                
        return moe_out, gate_probs

class EndometriosisPINN(nn.Module):
    """
    Physics-Informed Neural Network with Mixture of Experts (MoE) Architecture.
    """
    def __init__(self, feature_dim=160, hidden_dim=128, num_experts=4, k=2):
        super(EndometriosisPINN, self).__init__()
        
        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Mixture of Experts (MoE) Core Routing Network
        self.moe_layer = MoELayer(hidden_dim, num_experts, k)
        
        # Progression Predictor
        self.progression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Stage Classifier
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
        
        # Future Forecaster
        self.future_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3), 
            nn.Sigmoid()
        )
        
    def forward(self, combined_weighted_features):
        x = self.input_layer(combined_weighted_features)
        
        # Route through Mixture of Experts
        shared_rep, gate_probs = self.moe_layer(x)
        
        prob = self.progression_head(shared_rep)
        stage_logits = self.stage_classifier(shared_rep)
        future_risk = self.future_predictor(shared_rep)
        
        return prob, stage_logits, future_risk, gate_probs

    def physics_informed_loss(self, prob_pred, time_t, estradiol, ca125):
        """
        Calculates a physics-informed loss penalty.
        Endometriosis progression is often modeled as estrogen-dependent and
        correlates with inflammatory markers like CA-125.
        
        Constraint: d(Progression)/dt ~ alpha * Estradiol + beta * CA125
        Since we have static data, we simulate this constraint as a regularization term
        where predicted probability should monotonically align with high biological markers.
        """
        # A simple proxy differential constraint: 
        # higher estrogen & CA125 should strongly push the probability > 0.5
        expected_trend = (estradiol / 400.0) + (ca125 / 100.0) # Normalized proxy
        
        # Penalty if high markers don't result in high probability prediction
        penalty = torch.relu(0.8 * expected_trend - prob_pred.squeeze()) 
        return torch.mean(penalty ** 2)

class FullFedPINNModel(nn.Module):
    def __init__(self, ffnn_model=None, pinn_model=None, clinical_dim=9, us_dim=128, genomic_dim=256, path_dim=64, sensor_dim=32):
        super(FullFedPINNModel, self).__init__()
        from models.ffnn_weighting import FeatureWeightingFFNN
        self.ffnn = ffnn_model if ffnn_model else FeatureWeightingFFNN(clinical_dim, us_dim, genomic_dim, path_dim, sensor_dim)
        self.pinn = pinn_model if pinn_model else EndometriosisPINN(feature_dim=160)
        
    def forward(self, clinical_data, us_data, genomic_data, path_data, sensor_data):
        c_feat, u_feat, g_feat, p_feat, s_feat, weights = self.ffnn(clinical_data, us_data, genomic_data, path_data, sensor_data)
        
        # Apply attentive weights 
        weighted_c = c_feat * weights[:, 0:1]
        weighted_u = u_feat * weights[:, 1:2]
        weighted_g = g_feat * weights[:, 2:3]
        weighted_p = p_feat * weights[:, 3:4]
        weighted_s = s_feat * weights[:, 4:5]
        
        combined_weighted_features = torch.cat([weighted_c, weighted_u, weighted_g, weighted_p, weighted_s], dim=1)
        
        prob, stage_logits, future_risk, gate_probs = self.pinn(combined_weighted_features)
        return prob, stage_logits, future_risk, gate_probs

if __name__ == "__main__":
    from models.ffnn_weighting import FeatureWeightingFFNN
    
    ffnn = FeatureWeightingFFNN()
    pinn = EndometriosisPINN()
    full_model = FullFedPINNModel(ffnn, pinn)
    
    c_dummy = torch.randn(10, 9)
    u_dummy = torch.randn(10, 128)
    
    # We need all 5 streams to test the Full model
    g_dummy = torch.randn(10, 256)
    p_dummy = torch.randn(10, 64)
    s_dummy = torch.randn(10, 32)
    
    prob, stage, fut, gates = full_model(c_dummy, u_dummy, g_dummy, p_dummy, s_dummy)
    print("Probability Output:", prob.shape)
    print("Stage Logits:", stage.shape)
    print("MoE Gate Probs:", gates.shape)
    
    loss_phy = pinn.physics_informed_loss(prob, None, c_dummy[:, 7], c_dummy[:, 6])
    print("Physics Loss:", loss_phy.item())
