import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

from ..abstractions import AbstractAlgorithm


class ODE(AbstractAlgorithm):
    def __init__(self, 
                 embed_dim: int, 
                 timesteps: int, 
                 max_control_cost: float, 
                 lr=3e-2, 
                 seed=None,
                 **kwargs):
        
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        # ToDo: make layer sizes adapted to the embed_dim
        # ToDo: make lr adapted to embed_dim
        if seed:
            torch.manual_seed(seed)

        self.model = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), nn.Linear(32, embed_dim))
        self.lr = lr
        self.mse_loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)
    
    def forward(self, t, state):
        dx = self.model(state)
        return dx


    def fit(self, x: np.ndarray, epochs=100, method='rk4', **kwargs):
        x = torch.tensor(x, dtype=torch.float32)
        state = x[:, 0, :]
        t = torch.linspace(0.0, self._timesteps, self._timesteps)
        losses = []   
        for epoch in range(epochs):
            self.opt.zero_grad()
            pred = odeint(self.forward, state, t, method=method)
            pred = pred.transpose(0, 1)
            loss = self.mse_loss(pred, x).float()
            loss.backward()
            losses.append(loss.item())
            if epoch % 50 == 0 and kwargs.get('verbose', False):
                print(f"Epoch {epoch}, Loss {loss.item()}")
            self.opt.step()
        return losses
    
    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        x0 = torch.tensor(x0, dtype=torch.float32)
        t = torch.linspace(0.0, timesteps, timesteps)
        return odeint(self.forward, x0, t).transpose(0, 1).detach().numpy()
