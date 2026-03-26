import math
import torch
import torch.nn as nn

class PIMO(nn.Module):
    def __init__(
        self,
        num_genes,
        num_pathways,
        num_channels1,
        dropout1,
        num_channels2,
        dropout2,
        activation_fn,
        fc_nodes,
        pathway_mask,
    ):
        super().__init__()

        self.num_genes = num_genes
        self.num_pathways = num_pathways
        self.pathway_mask = pathway_mask
        self.num_channels1 = num_channels1
        self.num_channels2 = num_channels2
        self.fc_nodes = fc_nodes

        if activation_fn == "relu":
            activation_cls = nn.ReLU
        elif activation_fn == "sigmoid":
            activation_cls = nn.Sigmoid
        elif activation_fn == "tanh":
            activation_cls = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        self.stdv = 1.0 / math.sqrt(self.num_pathways * self.num_genes)

        self.W_Rk1 = nn.Parameter(
            torch.empty(self.num_pathways, self.num_genes).uniform_(-self.stdv, self.stdv) * self.pathway_mask
        )
        self.W_Rk2 = nn.Parameter(
            torch.empty(self.num_pathways, self.num_genes).uniform_(-self.stdv, self.stdv) * self.pathway_mask
        )
        self.W_Dq = nn.Parameter(
            torch.empty(self.num_pathways, self.num_genes).uniform_(-self.stdv, self.stdv) * self.pathway_mask
        )
        self.W_Cq = nn.Parameter(
            torch.empty(self.num_pathways, self.num_genes).uniform_(-self.stdv, self.stdv) * self.pathway_mask
        )

        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_channels1,
            kernel_size=(1, self.num_genes * 3),
            stride=(1, 1),
            padding=(0, 0),
        )

        self.hidden_activation = activation_cls()
        self.dropout1 = nn.Dropout(p=dropout1)

        self.conv2d_1 = nn.Conv2d(
            in_channels=self.num_channels1,
            out_channels=self.num_channels2,
            kernel_size=(2, 1),
            stride=(1, 1),
            padding="same",
        )

        self.hidden_activation1 = activation_cls()
        self.linear1 = nn.Linear(self.num_pathways * self.num_channels2, self.fc_nodes)
        self.linear1_activation = activation_cls()
        self.dropout2 = nn.Dropout(p=dropout2)

        self.linear2 = nn.Linear(self.fc_nodes, 1, bias=False)
        self.linear2.weight.data.uniform_(-0.001, 0.001)

    def forward(self, rna_data, dna_data, cna_data):
        Dq = dna_data * self.W_Dq
        Rk1 = rna_data * self.W_Rk1
        Dq_Rk = Dq * Rk1

        Cq = cna_data * self.W_Cq
        Rk2 = rna_data * self.W_Rk2
        Cq_Rk = Cq * Rk2

        Rv = rna_data

        deepMOR = torch.stack((Rv, Dq_Rk, Cq_Rk), dim=-1)
        deepMOR = deepMOR.view(deepMOR.shape[0], deepMOR.shape[1], -1).unsqueeze(1)

        out = self.conv2d(deepMOR)
        out = self.hidden_activation(out).squeeze(-1)
        out = self.dropout1(out)

        out = out.reshape(out.shape[0], self.num_channels1, self.num_pathways, 1)
        out = self.conv2d_1(out)
        out = self.hidden_activation1(out)

        out = out.view(out.shape[0], -1)
        out = self.dropout2(self.linear1_activation(self.linear1(out)))

        lin_pred = self.linear2(out)
        return lin_pred