import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Sequence, Tuple, Optional

from .base import DynamicsLearner


# ----- Simple MLP with Dropout -----
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 hidden: Sequence[int] = (128, 128, 64),
                 dropout: float = 0.15):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class ResidualNN(DynamicsLearner):
    """
    Neural-network residual learner with MC-dropout.
    predict() returns (mean, var) using MC-dropout at inference.
    Assumes caller handles any normalization.
    """

    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 r_dim: int,
                 hidden: Sequence[int] = (256, 256, 128),
                 lr: float = 5e-4,
                 dropout: float = 0.8,
                 weight_decay: float = 1e-4,
                 grad_clip: float = 0.3,
                 device: Optional[str] = None):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.r_dim = r_dim
        self.in_dim = x_dim + u_dim
        self.device = torch.device(
            device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        self.model = MLP(self.in_dim, r_dim, hidden=hidden, dropout=dropout).to(self.device)
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min',
                                                          factor=0.5, patience=5)
        # Huber is more robust than MSE for small/real datasets
        self.criterion = nn.HuberLoss(delta=1.0)
        self.grad_clip = grad_clip
        self._mc_passes_default = 50  # better variance estimate

        self.lambda_smooth = 1e-2
    def predict(self, x: Sequence[float], u: Sequence[float],
                mc_passes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict residual and epistemic uncertainty via MC-dropout.
        NOTE: We enable dropout (train mode) to sample predictive distribution.
        """
        mc_passes = mc_passes or self._mc_passes_default
        self.model.train()  # keep dropout on intentionally

        inp = np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]).astype(np.float32)
        xt = torch.from_numpy(inp).unsqueeze(0).to(self.device)

        preds = []
        with torch.no_grad():
            for _ in range(mc_passes):
                out = self.model(xt)
                preds.append(out.cpu().numpy().ravel())
        preds = np.stack(preds, axis=0)

        mean = preds.mean(axis=0)
        var = preds.var(axis=0) + 1e-8
        return mean, var

    def update(self,
               batch,
               epochs: int = 400,
               batch_size: int = 256,
               patience: int = 60) -> None:
        """
        Train on a provided batch [(x,u,r), ...] with early stopping.
        Assumes inputs/targets are already normalized outside.
        """
        if len(batch) == 0:
            return

        X, Y = [], []
        for x, u, r in batch:
            X.append(np.concatenate([np.asarray(x).ravel(), np.asarray(u).ravel()]))
            Y.append(np.asarray(r).ravel())
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        # small validation slice from the end (keeps chronology if caller passed in order)
        n = len(X)
        n_val = max(1, int(0.15 * n))
        X_tr, Y_tr = X[:-n_val], Y[:-n_val]
        X_val, Y_val = X[-n_val:], Y[-n_val:]

        tr_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=min(batch_size, len(tr_ds)),
                                                shuffle=True)

        X_val_t = torch.from_numpy(X_val).to(self.device)
        Y_val_t = torch.from_numpy(Y_val).to(self.device)

        best_val = float('inf')
        best_state = None
        patience_ctr = 0

        for _ in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in tr_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = self.criterion(pred, yb)

                self.opt.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.opt.step()

                epoch_loss += loss.item()
            epoch_loss /= max(1, len(tr_loader))

            # validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = self.criterion(val_pred, Y_val_t).item()

            self.sched.step(val_loss)

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()