import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

class simpleMLP(nn.Module):
    def __init__(self, n_input, n_output, hidden_dims=[64,64], 
        activation_fn = nn.Tanh, output_activation = None):
        super(simpleMLP,self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        
        layer_dims = [n_input] + hidden_dims + [n_output]
        layers = []

        for d in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[d], layer_dims[d+1]))
            if d < len(layer_dims) - 2:
                layers.append(activation_fn())
        
        if output_activation is not None:
            layers.append(output_activation())

        self.model = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.model(inputs)

class vanilla_model(nn.Module):
    def __init__(self, n_input, feature_dim=20, feat_hidden=[64,64], activation_fn=nn.Tanh,
        feat_activation=None, output_hidden=[64,64],output_activation=None,
        pred_Fz=True, pred_Fxy=False):
        super(vanilla_model, self).__init__()
        self.n_input = n_input
        self.n_output = 2 + int(pred_Fz) + 2*int(pred_Fxy)
        self.feature_dim = feature_dim
        self.feat_model = simpleMLP(n_input=n_input, n_output=feature_dim, 
            hidden_dims=feat_hidden, activation_fn=activation_fn, 
            output_activation=feat_activation
        )
        self.output_model = simpleMLP(feature_dim, self.n_output, hidden_dims=output_hidden,
            activation_fn=activation_fn, output_activation=output_activation)

    def forward(self, sens):
        return self.output_model(self.get_feature(sens))

    def get_feature(self, sens):
        return self.feat_model(sens)
    
    def get_out_from_feature(self, feature):
        return self.output_model(feature)