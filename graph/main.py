#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ExplanationEvaluation.explainers.AMExplainer import AMExplainer

from ExplanationEvaluation.models.GNN_paper import NodeGCN
# from ExplanationEvaluation.models.GNN_paper import NodeGCN_ba_community as NodeGCN
from ExplanationEvaluation.models.GNN_paper import GraphGCN

from ExplanationEvaluation.datasets.dataset_loaders import load_dataset

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)
parser.add_argument('--gamma', type=float)
parser.add_argument('--interval', type=int)
parser.add_argument('--slope_rate', type=float)
parser.add_argument('--move_rate', type=float)
parser.add_argument('--graph_id', type=int)

args_in = parser.parse_args()

#----------------------------------------------------------------------------------
graphs, features, labels, _, _, _ = load_dataset(args_in.dataset)
features = torch.tensor(features)
labels = torch.tensor(labels)

# the number of input dimension needs to be consistent with the definition of GNN model.
model = GraphGCN(10, args_in.num_classes)
path = f"./trained_gnn_models/GNN/{args_in.dataset}/best_model"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
#----------------------------------------------------------------------------------
task = args_in.task
explainer = AMExplainer(model, graphs, features, task, args_in.num_classes, args_in.epochs, args_in.lr, args_in.alpha, args_in.beta, args_in.gamma, args_in.interval, args_in.slope_rate, args_in.move_rate)
#----------------------------------------------------------------------------------
idx = args_in.graph_id
graph, expl, expl_slope = explainer.explain(idx)