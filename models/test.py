import torch
import torch.nn as nn
import numpy as np

from models import vanilla_model

def load_xyFz_model_and_scaling():
    model = vanilla_model(15, feature_dim=40, feat_hidden=[200,200], activation_fn=nn.ReLU,
                            feat_activation=None, output_hidden=[200,200],
                            output_activation=nn.ReLU)
    
    input_scaling = np.loadtxt('./input_scaling.txt')
    output_scaling = np.array([1./16, 1./16, 1/3.])
    model.load_state_dict(torch.load('./weights'))

    return model, input_scaling, output_scaling

if __name__ =='__main__':
    load_xyFz_model_and_scaling()
    # Inputs to the model must be (Change in Magnetic Field/input_scaling)
    # Model output must be scaled as (output/output_scaling). Units are (mm,mm,N).