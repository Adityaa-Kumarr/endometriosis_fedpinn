import shap
import torch
import numpy as np

class EndometriosisExplainer:
    """
    Explainable AI module using SHAP to interpret the 
    clinical and hormonal features leading to the Endometriosis prediction.
    """
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names or ['Age', 'BMI', 'Pelvic Pain', 'Dysmenorrhea', 
                                               'Dyspareunia', 'Family History', 'CA-125', 
                                               'Estradiol', 'Progesterone']
        
    def _model_wrapper(self, clinical_data):
        """
        Wrapper function for SHAP that takes numpy clinical data,
        adds dummy embeddings for other modalities (we mainly explain clinical/hormonal),
        and returns the model probabilities.
        """
        self.model.eval()
        tensor_data = torch.tensor(clinical_data, dtype=torch.float32)
        n = tensor_data.shape[0]
        us_data = torch.zeros((n, 128), dtype=torch.float32)
        genomic_data = torch.zeros((n, 256), dtype=torch.float32)
        path_data = torch.zeros((n, 64), dtype=torch.float32)
        sensor_data = torch.zeros((n, 32), dtype=torch.float32)
        with torch.no_grad():
            prob, _, _, _ = self.model(tensor_data, us_data, genomic_data, path_data, sensor_data)
        return prob.numpy()
    
    def explain_instance(self, background_data, instance_data, nsamples=100):
        """
        Generates SHAP values for a given instance using a background dataset.
        """
        # We use KernelExplainer for agnostic models
        explainer = shap.KernelExplainer(self._model_wrapper, background_data)
        shap_values = explainer.shap_values(instance_data, nsamples=nsamples)
        
        return explainer, shap_values

    def plot_summary(self, explainer, shap_values, instance_data):
        """
        Generates a summary plot. In a real application, this would save
        an image or return a matplotlib figure to Streamlit.
        """
        shap.summary_plot(shap_values, instance_data, feature_names=self.feature_names, show=False)

if __name__ == "__main__":
    from models.ffnn_weighting import FeatureWeightingFFNN
    from models.pinn import EndometriosisPINN, FullFedPINNModel

    ffnn = FeatureWeightingFFNN()
    pinn = EndometriosisPINN()
    full_model = FullFedPINNModel(ffnn, pinn)

    explainer = EndometriosisExplainer(full_model)
    bg_data = np.random.randn(10, 9)
    inst_data = np.random.randn(2, 9)
    
    exp, vals = explainer.explain_instance(bg_data, inst_data)
    print("SHAP Values Shape:", np.array(vals).shape)
