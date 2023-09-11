import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torch.nn.parameter import Parameter
import numpy as np
from torch.distributions import Normal


class Model(nn.Module):
    def __init__(self, lstm_units=4, seqlen=5):
        super(Model, self).__init__()
        self.seqlen = seqlen
        self.lstm_units = lstm_units
        self.batch_size = 1
        self.veldim = 6

        init_mu = 0.12
        init_sigma = 0.10
        self.n_sample = 5
        self.n_elite = 1

        ### Initialise Mu and Sigma
        # we change 1,1 to 1,6
        self.mu = init_mu * torch.ones((1, 6), requires_grad=True).cuda()
        self.sigma = init_sigma * torch.ones((1, 6), requires_grad=True).cuda()
        self.dist = Normal(self.mu, self.sigma)

        self.f_interm = []
        self.v_interm = []
        self.mean_interm = []

        self.block = nn.Sequential(
            nn.Linear(6, 256),
            nn.Sigmoid(),
            nn.Dropout(0.6),
        )

        # we add linear layer instead of unblocking layer

        self.linear = nn.Linear(256, 6)
        self.pooling = torch.nn.AvgPool2d(kernel_size=1, stride=1)

    def forward(self, vel, Lsx, Lsy, horizon, f12):
        vel = self.dist.rsample((self.n_sample,)).cuda()
        # n x 6 velocity
        # vel = torch.rand(8, 1, device=torch.device('cuda:0'))
        # we change (8,1,1) to (8,1,6)
        self.f_interm.append(self.sigma)
        self.mean_interm.append(self.mu)
        vel = self.block(vel.view(self.n_sample, 1, 6))
        vel = torch.sigmoid(self.linear(vel)).view(self.n_sample, 6)
        # 8, 6 Velocity Vector

        # self.mean_interm.append(torch.mean(vel, dim=0))

        vel = vel.view(self.n_sample, 1, 1, 6) * 2 - 1
        ### Horizon Bit
        if horizon < 0:
            flag = 0
        else:
            flag = 1

        if flag == 1:
            vels = vel * horizon
        else:
            vels = vel * -horizon

        Lsx = Lsx.view(1, f12.shape[2], f12.shape[3], 6)
        Lsy = Lsy.view(1, f12.shape[2], f12.shape[3], 6)
        # print("Lsx Shape, Lsy Shape: ", Lsx.shape, Lsy.shape)

        f_hat = torch.cat(
            (
                torch.sum(Lsx * vels, -1).unsqueeze(-1),
                torch.sum(Lsy * vels, -1).unsqueeze(-1),
            ),
            -1,
        )

        f_hat = self.pooling(f_hat.permute(0, 3, 1, 2))
        if horizon < 0:
            loss_fn = torch.nn.MSELoss(size_average=False, reduce=False)
            # print("Fat size:", f_hat.size())
            # print("F12 size:", f12.size())
            loss = loss_fn(f_hat, f12)
            loss = torch.mean(loss.reshape(self.n_sample, -1), dim=1)
            sorted, indices = torch.sort(loss)
            loss_norm = torch.softmax(sorted[0], 0)
            vel = vel[indices[0]]
            vel = vel * loss_norm
            vel = vel.view(
                6,
            )
            self.v_interm.append(vel)

        # 1 x 6
        # 1 x 1
        # variational inferencing.
        # 1 dist : mu, sigma
        #
        # half cem: one
        # update sigma
        mu_copy = self.mu.detach().clone()
        self.mu = vel  # mu, sigma
        self.sigma = ((mu_copy - self.mu) ** 2).sqrt()
        # ((I * (X - mu.unsqueeze(1))**2).sum(dim=1) / n_elite).sqrt()
        return f_hat


if __name__ == "__main__":
    vs = Model().to("cuda:0")
    ve = torch.zeros(6).to("cuda:0")
    print(list(vs.parameters()))