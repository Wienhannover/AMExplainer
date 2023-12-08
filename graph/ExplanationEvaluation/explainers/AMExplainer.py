import torch
import torch_geometric as ptgeom
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from tqdm import tqdm
from math import sqrt
import numpy as np

from ExplanationEvaluation.explainers.BaseExplainer import BaseExplainer
from ExplanationEvaluation.utils.graph import index_edge


class AMExplainer(BaseExplainer):

    def __init__(self, model_to_explain, graphs, features, task, num_classes, epochs=30, lr=0.003, alpha=1.0, beta=1.0, gamma=1.0, interval=1000, slope_rate=1, move_rate=1):
        super().__init__(model_to_explain, graphs, features, task)

        self.num_classes = num_classes
        self.epochs = epochs
        self.lr = lr
        self.alpha = alpha
        self.beta= beta
        self.gamma = gamma
        self.interval = interval
        self.slope_rate = slope_rate
        self.move_rate = move_rate

        self.all_loss = []
        self.loss_1 = []
        self.loss_2 = []
        self.loss_3 = []
        
        #-------------------------------------
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)
        #-------------------------------------

    def _set_masks(self, x, edge_index):

        (N, F), E = x.size(), edge_index.size(1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)


    def _loss(self, masked_pred, masked_pred_random, original_pred, masked_emb, original_emb, a, b, c):

        mse_1 = torch.sum((masked_pred-original_pred)**2)
        
        uniform_dis = torch.ones_like(masked_pred_random) * (1/self.num_classes)        
        mse_2 = torch.sum((self.softmax(masked_pred_random)-uniform_dis)**2)

        # ignore this part with c=0.
        mse_3 = torch.sum((masked_emb-original_emb)**2)
                
        print(mse_1, mse_2, mse_3)

        return a*mse_1 + b*mse_2 + c*mse_3, mse_1, mse_2, mse_3
        

    def sigmoid_slope(self, x, rate, move_distance):

        return 1 / (1 + torch.exp(- rate*(x-move_distance)))


    def explain(self, index):

        index = int(index)

        # Prepare model for new explanation run
        self.model_to_explain.eval()
        
        
        if self.type == 'node':
            # Similar to the original paper we only consider a subgraph for explaining
            feats = self.features
            graph = ptgeom.utils.k_hop_subgraph(index, 3, self.graphs)[1]
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)[index]

                original_emb = self.model_to_explain.embedding(feats, graph)[index]

                pred_label = original_pred.argmax(dim=-1).detach()
        else:
            #***************************************
            graph = torch.tensor(self.graphs[index])
            #***************************************
            feats = self.features[index].detach()
            #graph = self.graphs[index].detach()
            graph = graph.detach()

            # Remove self-loops
            graph = graph[:, (graph[0] != graph[1])]
            with torch.no_grad():
                original_pred = self.model_to_explain(feats, graph)
                original_emb = self.model_to_explain.embedding(feats, graph)
                pred_label = original_pred.argmax(dim=-1).detach()
                
        #-----------------------------------------------------------------------------------
        self._set_masks(feats, graph)
        #-----------------------------------------------------------------------------------
        optimizer = Adam([self.edge_mask], lr=self.lr)
        
        a = self.alpha
        b = self.beta
        c = self.gamma
                
        self.all_loss = []
        self.loss_1 = []
        self.loss_2 = []
        self.loss_3 = []


        move_distance = 0
        curr_slope = 1

        for e in range(1, self.epochs):
            
            if (e %self.interval) == 0:

                move_distance = move_distance + self.move_rate
                curr_slope = curr_slope + self.slope_rate
            
            #-----------------------------------------------------------------------------------  
            print("------------------------------------", e)
            print("------------------------------------", a, b, c)
            print("--------------slope of sigmoid------", curr_slope)
            print("--------------move distance----------", move_distance)
 
            optimizer.zero_grad()

            if self.type == 'node':
                
                #-----------------------------------------------------------------------------------
                masked_pred = self.model_to_explain(feats, graph, edge_weights=self.sigmoid_slope(self.edge_mask, curr_slope, move_distance))[index]

                # ignore this part with c=0.
                masked_emb = self.model_to_explain.embedding(feats, graph, edge_weights=self.sigmoid_slope(self.edge_mask, curr_slope, move_distance))[index]

                masked_pred_random = self.model_to_explain(feats, graph, edge_weights=(1-self.sigmoid_slope(self.edge_mask, curr_slope, move_distance)))[index]
                #-----------------------------------------------------------------------------------
                loss, loss_1, loss_2, loss_3= self._loss(masked_pred.unsqueeze(0), masked_pred_random.unsqueeze(0), original_pred.unsqueeze(0), masked_emb, original_emb, a, b, c)
                
                self.all_loss.append(loss.detach().item())
                self.loss_1.append(loss_1.detach().item())
                self.loss_2.append(loss_2.detach().item())
                self.loss_3.append(loss_3.detach().item())

            else:
                #-----------------------------------------------------------------------------------
                masked_pred = self.model_to_explain(feats, graph, edge_weights=self.sigmoid_slope(self.edge_mask, curr_slope, move_distance))

                masked_emb = self.model_to_explain.embedding(feats, graph, edge_weights=self.sigmoid_slope(self.edge_mask, curr_slope, move_distance))

                masked_pred_random = self.model_to_explain(feats, graph, edge_weights=(1-self.sigmoid_slope(self.edge_mask, curr_slope, move_distance)))
                #-----------------------------------------------------------------------------------
                loss, loss_1, loss_2, loss_3= self._loss(masked_pred, masked_pred_random, original_pred, masked_emb, original_emb, a, b, c)
                
                self.all_loss.append(loss.detach().item())
                self.loss_1.append(loss_1.detach().item())
                self.loss_2.append(loss_2.detach().item())
                self.loss_3.append(loss_3.detach().item())

            loss.backward()
            print(loss.detach().item())
            optimizer.step()

        print("final loss", loss.detach().item())

        if self.type == 'node':

            self.edge_mask = self.edge_mask.detach()

            for tmp_index in range(1, self.edge_mask.shape[0]+1):

                # tmp_edge_mask = self.edge_mask.clone()
                tmp_edge_mask = self.sigmoid_slope(self.edge_mask, curr_slope, move_distance).clone()

                tmp_kth_index = torch.topk(tmp_edge_mask, tmp_index)[1][-1].item()
                tmp_kth_value = torch.topk(tmp_edge_mask, tmp_index)[0][-1].item()

                corresponding_edge = (graph[0][tmp_kth_index].item(), graph[1][tmp_kth_index].item())

                tmp_new_mask = (tmp_edge_mask >= tmp_kth_value) * tmp_edge_mask
                tmp_masked_pred = self.model_to_explain(feats, graph, edge_weights=tmp_new_mask)[index]
                print(f"{tmp_index}th value---------", tmp_masked_pred.detach(), tmp_kth_value, corresponding_edge)

        else:
            self.edge_mask = self.edge_mask.detach()

            for tmp_index in range(1, self.edge_mask.shape[0]+1):

                # tmp_edge_mask = self.edge_mask.clone()
                tmp_edge_mask = self.sigmoid_slope(self.edge_mask, curr_slope, move_distance).clone()

                tmp_kth_index = torch.topk(tmp_edge_mask, tmp_index)[1][-1].item()
                tmp_kth_value = torch.topk(tmp_edge_mask, tmp_index)[0][-1].item()

                corresponding_edge = (graph[0][tmp_kth_index].item(), graph[1][tmp_kth_index].item())

                tmp_new_mask = (tmp_edge_mask >= tmp_kth_value) * tmp_edge_mask
                tmp_masked_pred = self.model_to_explain(feats, graph, edge_weights=tmp_new_mask)
                print(f"{tmp_index}th value---------", tmp_masked_pred.detach(), tmp_kth_value, corresponding_edge)


        # Retrieve final explanation
        mask = torch.sigmoid(self.edge_mask)
        expl_graph_weights = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)): # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights[t] = mask[i]


        # Retrieve final explanation
        print("$$$$$$$$$$$$$$$$   final slope           $$$$$$$$$$$$$$$$$$")
        print(curr_slope)
        print("$$$$$$$$$$$$$$$$   final move distance   $$$$$$$$$$$$$$$$$$")
        print(move_distance)
        mask_slope = self.sigmoid_slope(self.edge_mask, curr_slope, move_distance)
        expl_graph_weights_slope = torch.zeros(graph.size(1))
        for i in range(0, self.edge_mask.size(0)): # Link explanation to original graph
            pair = graph.T[i]
            t = index_edge(graph, pair)
            expl_graph_weights_slope[t] = mask_slope[i]


        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(self.edge_mask)
        print(expl_graph_weights)
        print(expl_graph_weights_slope)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        if self.type == 'node':

            #-----------------------------------------------------------------------------------
            masked_pred = self.model_to_explain(feats, graph, edge_weights=self.sigmoid_slope(self.edge_mask, curr_slope, move_distance))[index]
            masked_pred_random = self.model_to_explain(feats, graph, edge_weights=(1-self.sigmoid_slope(self.edge_mask, curr_slope, move_distance)))[index]
            
            print("original prediction", original_pred, self.softmax(original_pred), "predicted label", pred_label)
            print("masked prediction", masked_pred.detach(), self.softmax(masked_pred.detach()))
            print("random prediction", masked_pred_random.detach(), self.softmax(masked_pred_random.detach()))
            #-----------------------------------------------------------------------------------

            print("---------------------------------------------")
            tmp_edge_mask = self.edge_mask.clone()
            tmp_weight = self.sigmoid_slope(tmp_edge_mask, curr_slope, move_distance)
            min_val = 0.1
            tmp_new_weight = (tmp_weight > min_val) * tmp_weight
            tmp_masked_pred = self.model_to_explain(feats, graph, edge_weights=tmp_new_weight)[index]
            print("final weights")
            print(tmp_new_weight)
            print(f"number of edges larger than min_val {min_val}: ",  (tmp_weight > min_val).sum().item(), tmp_masked_pred.detach(), self.softmax(tmp_masked_pred.detach()))
            print("---------------------------------------------")
            tmp_zeros_weight = torch.zeros_like(self.edge_mask)
            tmp_zeros_pred = self.model_to_explain(feats, graph, edge_weights=tmp_zeros_weight)[index]
            print("prediction of zeros mask", tmp_zeros_pred.detach(), self.softmax(tmp_zeros_pred.detach()))
            print("---------------------------------------------")

        else:

            #-----------------------------------------------------------------------------------
            masked_pred = self.model_to_explain(feats, graph, edge_weights=self.sigmoid_slope(self.edge_mask, curr_slope, move_distance))
            masked_pred_random = self.model_to_explain(feats, graph, edge_weights=(1-self.sigmoid_slope(self.edge_mask, curr_slope, move_distance)))
            
            print("original prediction", original_pred, self.softmax(original_pred), "predicted label", pred_label)
            print("masked prediction", masked_pred.detach(), self.softmax(masked_pred.detach()))
            print("random prediction", masked_pred_random.detach(), self.softmax(masked_pred_random.detach()))
            #-----------------------------------------------------------------------------------

            print("---------------------------------------------")
            tmp_edge_mask = self.edge_mask.clone()
            tmp_weight = self.sigmoid_slope(tmp_edge_mask, curr_slope, move_distance)
            min_val = 0.1
            tmp_new_weight = (tmp_weight > min_val) * tmp_weight
            tmp_masked_pred = self.model_to_explain(feats, graph, edge_weights=tmp_new_weight)
            print("final weights")
            print(tmp_new_weight)
            print(f"number of edges larger than min_val {min_val}: ",  (tmp_weight > min_val).sum().item(), tmp_masked_pred.detach(), self.softmax(tmp_masked_pred.detach()))
            print("---------------------------------------------")


        return graph, expl_graph_weights, expl_graph_weights_slope


