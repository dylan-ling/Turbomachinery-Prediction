import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv
from torch_geometric.nn.pool import knn_graph

class DGCNNReg(nn.Module):
    def __init__(self, k=20, output_dim=4):
        super().__init__()
        self.k = k
        in_channels = 13
        # helper MLP builder 
        def mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c,  out_c),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(out_c, out_c),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.conv1 = EdgeConv(mlp(2 * in_channels,   64), aggr='max')
        self.conv2 = EdgeConv(mlp(128, 64), aggr='max')
        self.conv3 = EdgeConv(mlp(128,128), aggr='max')
        self.conv4 = EdgeConv(mlp(256,256), aggr='max')
        self.conv5 = EdgeConv(mlp(512,128), aggr='max')

        self.mlp = nn.Sequential(
            nn.Linear(64 + 64 + 128 + 256 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim),
        )

    def forward(self, data):
        x, batch = data.x, data.batch
        coords = x[:, :3]

        # 3) Build static kNN once per forward
        edge_index = knn_graph(x, k=self.k, batch=batch, loop=False)

        # 4) Pass that same edge_index to all EdgeConv layers
        x1 = self.conv1(x, edge_index)        #  [sum(N),64]
        x2 = self.conv2(x1, edge_index)       #  [sum(N),64]
        x3 = self.conv3(x2, edge_index)       #  [sum(N),128]
        x4 = self.conv4(x3, edge_index)       #  [sum(N),256]
        x5 = self.conv5(x4, edge_index)       #  [sum(N),128]

        # concatenate & MLP head
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)  # [sum(N),256]
        out   = self.mlp(x_cat)                 # [sum(N),4]
        return out
