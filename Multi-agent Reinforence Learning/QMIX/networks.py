#!/usr/bin/python3

# Author(s): Luiz Felipe Vecchietti, Kyujin Choi, Taeyoung Kim
# Maintainer: Kyujin Choi (nav3549@kaist.ac.kr)

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json

class Agent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Agent, self).__init__()
        params_file = open(os.path.dirname(__file__) + '/parameters.json')
        params = json.loads(params_file.read())
        self.hidden_dim = params['agent_parameters']['hidden_dim']
        self.input_linear = torch.nn.Linear(num_inputs, self.hidden_dim)
        self.middle_linear = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = torch.nn.Linear(self.hidden_dim, num_outputs)
        self.num_layers = params['agent_parameters']['num_layers']

    def forward(self, x):

        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(0, self.num_layers):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        q_pred = self.output_linear(h_relu)
        return q_pred

    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)