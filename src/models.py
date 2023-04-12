import torch as t
from torch.nn import GRU

class Att_GRU(t.nn.Module):
    def __init__(self, f_dim, n_layers, hidden, window, device='cpu'):
        super().__init__()

        self.f_dim = t.scalar_tensor(f_dim, device=device, dtype=t.int16)
        self.hidden = t.scalar_tensor(hidden, device=device, dtype=t.int16)

        self.att_1 = t.nn.Parameter(t.rand((1, f_dim, f_dim), dtype=t.float32, requires_grad=True)).to(device=device)
        self.att_2 = t.nn.Parameter(t.randn((1, hidden, hidden), dtype=t.float32, requires_grad=True)).to(device=device)
        
        self.sm = t.nn.Softmax(dim=1)

        self.gru = GRU(
            input_size = f_dim,
            hidden_size = hidden,
            num_layers = n_layers,
            bidirectional=False
        )

        self.w = t.nn.Sequential(
            t.nn.Linear(hidden * window, 512),
            t.nn.Linear(512, 3),
        ) 

    def forward(self, x):
        # first attention stage
        pr = self.att_1.repeat(x.shape[0],1,1)
        pr = t.bmm(x, pr)
        pr = (self.sm(pr).sum(2) / self.f_dim)\
            .unsqueeze(1)\
            .repeat(1, self.f_dim, 1)\
            .permute(0,2,1)
        x = pr * x

        # gru
        x, _ = self.gru(x)

        # second attention stage
        pr = self.att_2.repeat(x.shape[0],1,1)
        pr = t.bmm(x, pr)
        pr = (self.sm(pr).sum(2) / self.hidden)\
            .unsqueeze(1)\
            .repeat(1, self.hidden, 1)\
            .permute(0,2,1)
        
        x = (pr * x).flatten(1)
        x = self.w(x)
        x = self.sm(x)

        return x
