from torch_geometric.nn import GATConv
import torch.nn as nn
import torch
import torch.nn.functional as F


class GAT(nn.Module):
    def __init__(self, num_features, num_classes):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(
            self.hid * self.in_head,
            num_classes,
            concat=False,
            heads=self.out_head,
            dropout=0.6,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class HypGAT(nn.Module):
    """
    Dynamic GAT model that allows us to alter the architecture/hyperparameters 
    using config files or bayesian optimisation.

    See config/cora_default.json for an example definition.
    The main idea is to specify all the keyword arguments num_feature and num_classes will be passed in during training.
    """
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_channels=8,
        layer_heads=[8, 1],
        dropout=0.6,
        residual=False,
        layer_norm=False
    ):
        super().__init__()
        self.n_layers = len(layer_heads)
        self.dropout = dropout
        self.layer_norm = layer_norm

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        curr_in = num_features
        for heads in layer_heads[:-1]:
            # Hidden layer out dimension all set to hidden_channels
            self.gat_layers.append(
                GATConv(curr_in, hidden_channels // heads, heads=heads, dropout=self.dropout, residual=residual)
            )
            curr_in = hidden_channels

            # Add Layer Norm if selected
            if self.layer_norm:
                self.norm_layers.append(
                    nn.LayerNorm(curr_in)
                )

        # Last layer out dimension must match num_classes
        self.gat_layers.append(
            GATConv(
                curr_in,
                num_classes,
                heads=layer_heads[-1],
                concat=False,
                dropout=self.dropout,
                residual=residual
            )
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.gat_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)

            # Apply ELU unless last layer then apply softmax
            if i < self.n_layers - 1:
                x = F.relu(x)
                if self.layer_norm:
                    x = self.norm_layers[i](x)
            else:
                x = F.log_softmax(x, dim=1)

        return x

class HypGAT_PPI(nn.Module):
    """
    Dynamic GAT model for the PPI dataset that allows us to alter the architecture/hyperparameters 
    using config files or bayesian optimisation.

    For an exact match to GAT_own_PPI, set:
        hidden_channels=256
        layer_heads=[4, 4, 6]
        dropout=0.6
        residual=True
        concat=True (the final layer is forced to concat=False anyway).
    """
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_channels=8,
        layer_heads=[4,4,6],
        dropout=0.6,
        residual=True,
        concat=True
    ):
        super().__init__()
        self.n_layers = len(layer_heads)
        self.dropout = dropout
        self.residual = residual
        self.concat = concat

        gat_layers = []
        curr_in = num_features
        for heads in layer_heads[:-1]:
            # Intermediate GATConv layers
            gat_layers.append(
                GATConv(
                    in_channels=curr_in,
                    out_channels=hidden_channels,
                    heads=heads,
                    dropout=self.dropout,
                    residual=self.residual,
                    concat=self.concat
                )
            )
            curr_in = hidden_channels * heads

        # Final GATConv layer -> output dimension must match num_classes
        gat_layers.append(
            GATConv(
                in_channels=curr_in,
                out_channels=num_classes,
                heads=layer_heads[-1],
                concat=False,            # final layer does not concatenate heads
                dropout=self.dropout,
                residual=self.residual
            )
        )

        self.gat_layers = nn.ModuleList(gat_layers)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.gat_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)

            # Apply ELU to all but the last layer
            if i < self.n_layers - 1:
                x = F.elu(x)

        # Return raw logits (no sigmoid), since the binary cross-entropy loss includes it
        return x