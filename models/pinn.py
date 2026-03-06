import torch
import torch.nn as nn

class EndometriosisPINN(nn.Module):
    """
    Physics-Informed Neural Network for Endometriosis prediction.
    Integrates the weighted features from the FFNN and outputs progression stages.
    """
    def __init__(self, feature_dim=160, hidden_dim=128):
        super(EndometriosisPINN, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), # Predict disease progression proxy/probability
            nn.Sigmoid()
        )
        
        # Multi-class output for Endometriosis Stage (None, Minimal, Mild, Moderate, Severe) -> 5 classes
        self.stage_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 5)
        )
        
        # Future Risk Forecaster (outputs risk at 1-year, 3-years, 5-years)
        self.future_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 3), 
            nn.Sigmoid()
        )
        
    def forward(self, combined_weighted_features):
        # Extract activations before the final layer for stage classification
        x = combined_weighted_features
        for layer in self.predictor[:-2]: # Iterate until the last hidden layer
            x = layer(x)
            
        prob = self.predictor[-2:](x) # The sigmoid output
        stage_logits = self.stage_classifier(x)
        future_risk = self.future_predictor(x)
        
        return prob, stage_logits, future_risk

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
        
        prob, stage_logits, future_risk = self.pinn(combined_weighted_features)
        return prob, stage_logits, future_risk

if __name__ == "__main__":
    from ffnn_weighting import FeatureWeightingFFNN
    
    ffnn = FeatureWeightingFFNN()
    pinn = EndometriosisPINN()
    full_model = FullFedPINNModel(ffnn, pinn)
    
    c_dummy = torch.randn(10, 9)
    u_dummy = torch.randn(10, 128)
    
    prob, stage = full_model(c_dummy, u_dummy)
    print("Probability Output:", prob.shape)
    print("Stage Logits:", stage.shape)
    
    loss_phy = pinn.physics_informed_loss(prob, None, c_dummy[:, 7], c_dummy[:, 6])
    print("Physics Loss:", loss_phy.item())
