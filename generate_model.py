import torch
import sys
import os

from models.ffnn_weighting import FeatureWeightingFFNN
from models.pinn import EndometriosisPINN, FullFedPINNModel

def generate():
    ffnn = FeatureWeightingFFNN()
    pinn = EndometriosisPINN()
    model = FullFedPINNModel(ffnn, pinn)
    torch.save({'full_model': model.state_dict()}, 'global_model.pth')
    print("global_model.pth has been successfully created!")

if __name__ == "__main__":
    generate()
