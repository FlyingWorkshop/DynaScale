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
                 lr=2e-3, 
                 seed=None,
                 **kwargs):
        
        super().__init__(embed_dim, timesteps, max_control_cost, **kwargs)

        # ToDo: make layer sizes adapted to the embed_dim
        # ToDo: make lr adapted to embed_dim
        if seed:
            torch.manual_seed(seed)

        self.model = nn.Sequential(nn.Linear(embed_dim, 32), nn.Softplus(), 
                                   nn.Linear(32, 32), nn.Softplus(), nn.Linear(32, embed_dim))
        self.lr = lr
        self.mse_loss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-3) #1e-4

    def forward(self, t, state):
        dx = self.model(state)
        return dx
    
    def warmup(self, x: np.ndarray, lr=3e-1, epochs=20, method='implicit_adams', **kwargs):
        warm_opt = torch.optim.LBFGS(self.model.parameters(), lr=3e-1, tolerance_grad=1e-05)
        lr_fn = lambda epoch: (3e-1 - 1e-1) * epoch/50 + 1e-1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(warm_opt, lr_lambda=lr_fn)

        x = torch.tensor(x, dtype=torch.float32)
        state = x[:, 0, :]
        t = torch.linspace(0.0, self._timesteps, self._timesteps)
        losses = []
        epochs=50
        for epoch in range(epochs):
            def closure():
                warm_opt.zero_grad()
                pred = odeint(self.forward, state, t, method=method)
                pred = pred.transpose(0, 1)
                loss = self.mse_loss(pred, x).float()
                loss.backward()
                return loss
            
            warm_opt.zero_grad()
            pred = odeint(self.forward, state, t, method=method)
            pred = pred.transpose(0, 1)
            loss = self.mse_loss(pred, x).float()
            loss.backward()
            losses.append(loss.item())
            #if epoch % 5 == 0 and kwargs.get('verbose', False):
            #print(f"Epoch {epoch}, Loss {loss.item()}")
            warm_opt.step(closure)
            #lr_scheduler.step()
            if loss.item() < 10:
                break
        return losses


    def fit(self, x: np.ndarray, epochs=5000, method='dopri5', warmup=True, **kwargs):
        if warmup:
            self.warmup(x, lr=1)
            warm_lr = kwargs.get('warm_lr', 2e-4)
            warm_epochs = 0.1*2000
            lr_fn = lambda epoch: warm_lr + (self.lr - warm_lr) / (1 + np.exp(-epoch + warm_epochs/2))
        else:
            lr_fn = lambda epoch : self.lr
       
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_fn)
        x = torch.tensor(x, dtype=torch.float32)
        state = x[:, 0, :]
        t = torch.linspace(0.0, self._timesteps, self._timesteps)
        losses = []

        patience = 20
        delta = 0.1
        best_loss = None
        counter = 0
        early_stop = False

        for epoch in range(epochs):
            self.opt.zero_grad()
            pred = odeint(self.forward, state, t, method=method)
            pred = pred.transpose(0, 1)
            loss = self.mse_loss(pred, x).float()
            loss.backward()
            losses.append(loss.item())
            #if epoch % 5 == 0 and kwargs.get('verbose', False):
            #print(f"Epoch {epoch}, Loss {loss.item()}")
            self.opt.step()
            lr_scheduler.step()
            
            if best_loss is None:
                best_loss = loss.item()
            elif loss.item() > best_loss - delta:
                counter += 1
                if counter >= patience:
                    early_stop = True
                    break
            else:
                best_loss = loss.item()
                counter = 0
        return losses
    
    def predict(self, x0: np.ndarray, timesteps: int, **kwargs) -> np.ndarray:
        x0 = torch.tensor(x0, dtype=torch.float32)
        t = torch.linspace(0.0, timesteps, timesteps)
        return odeint(self.forward, x0, t).transpose(0, 1).detach().numpy()