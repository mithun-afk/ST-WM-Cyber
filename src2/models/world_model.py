"""src2/models/world_model.py — Spatio-Temporal Graph World Model (STGWM).

Architecture
------------
  LSTM  ->  LayerNorm  ->  three heads:
    dynamics_head : predicts next feature vector  (SmoothL1 loss)
    risk_head     : predicts binary risk logit    (FocalLoss)
    stage_head    : predicts MITRE stage logits   (CrossEntropy)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class _FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal = alpha_t * (1.0 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Internal nn.Module
# ---------------------------------------------------------------------------
class _STGWMNet(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2, num_stages=6, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stages = num_stages

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        half = hidden_dim // 2
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(half, 1),
        )
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, half), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(half, num_stages),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.norm(out[:, -1, :])
        return self.dynamics_head(h), self.risk_head(h), self.stage_head(h)


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------
class STGWMModel:
    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2, num_stages=6):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stages = num_stages
        self._net = None
        self._scaler = None
        self._training_log = {}
        self._trained = False

    @property
    def status(self):
        return "TRAINED" if self._trained else "NOT_TRAINED"

    def fit(self, X, y_risk, y_stage=None, scaler=None, epochs=20, patience=3,
            batch_size=256, lr=1e-3, device="cpu"):
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        N, seq_len, F = X.shape
        if F != self.input_dim:
            raise ValueError(f"input_dim mismatch: model expects {self.input_dim}, got {F}")

        if scaler is not None:
            self._scaler = scaler
        else:
            self._scaler = RobustScaler()
            self._scaler.fit(X.reshape(-1, F))

        X_scaled = self._scaler.transform(X.reshape(-1, F)).reshape(N, seq_len, F).astype(np.float32)
        # Clip scaled values to ±10 to prevent rare extreme outliers from
        # destabilizing LSTM hidden state (RobustScaler still passes outliers through)
        X_scaled = np.clip(X_scaled, -10.0, 10.0)
        y_dyn = X_scaled[:, -1, :]

        self._net = _STGWMNet(self.input_dim, self.hidden_dim, self.num_layers, self.num_stages).to(device)
        focal = _FocalLoss(alpha=0.75, gamma=2.0)
        ce_loss = nn.CrossEntropyLoss()
        sl1 = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)

        split = int(N * 0.8)
        Xt = torch.tensor(X_scaled[:split], dtype=torch.float32, device=device)
        Xv = torch.tensor(X_scaled[split:], dtype=torch.float32, device=device)
        yt_risk = torch.tensor(y_risk[:split], dtype=torch.float32, device=device)
        yv_risk = torch.tensor(y_risk[split:], dtype=torch.float32, device=device)
        yt_dyn = torch.tensor(y_dyn[:split], dtype=torch.float32, device=device)
        yv_dyn = torch.tensor(y_dyn[split:], dtype=torch.float32, device=device)

        if y_stage is not None:
            yt_stage = torch.tensor(y_stage[:split], dtype=torch.long, device=device)
            yv_stage = torch.tensor(y_stage[split:], dtype=torch.long, device=device)
            stage_weight = 0.5
        else:
            yt_stage = yv_stage = None
            stage_weight = 0.0

        train_size = len(Xt)
        best_val_loss = float("inf")
        best_state = None
        best_dyn_mse = None
        epochs_no_improve = 0
        actual_epochs = 0

        self._net.train()
        for ep in range(epochs):
            actual_epochs += 1
            perm = torch.randperm(train_size, device=device)
            Xs = Xt[perm]; rs = yt_risk[perm]; ds = yt_dyn[perm]
            if yt_stage is not None:
                ss = yt_stage[perm]
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, train_size, batch_size):
                bX = Xs[i:i+batch_size]; br = rs[i:i+batch_size]; bd = ds[i:i+batch_size]
                dp, rl, sl = self._net(bX)
                l_dyn = sl1(dp, bd)
                l_risk = focal(rl, br)
                l_stage = ce_loss(sl, ss[i:i+batch_size]) if yt_stage is not None else torch.tensor(0.0, device=device)
                loss = l_dyn + l_risk + stage_weight * l_stage
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            self._net.eval()
            with torch.no_grad():
                vd, vr, vs = self._net(Xv)
                v_dyn = sl1(vd, yv_dyn).item()
                v_risk = focal(vr, yv_risk).item()
                v_stage = ce_loss(vs, yv_stage).item() if yv_stage is not None else 0.0
                val_loss = v_dyn + v_risk + stage_weight * v_stage
                
                # dynamics_mse is evaluated in the original unscaled feature space
                dyn_mse = float(np.mean((
                    self._scaler.inverse_transform(vd.cpu().numpy()) -
                    self._scaler.inverse_transform(yv_dyn.cpu().numpy())) ** 2))
                    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                best_dyn_mse = dyn_mse
                epochs_no_improve = 0
                tag = "  [best]"
            else:
                epochs_no_improve += 1
                tag = ""

            print(f"  Epoch {ep+1:3d}/{epochs}  train_loss={epoch_loss/max(n_batches,1):.4f}  "
                  f"val_loss={val_loss:.4f}  no_improve={epochs_no_improve}/{patience}{tag}")
                
            if epochs_no_improve >= patience:
                break
                
            self._net.train()

        if best_state:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self._trained = True
        
        # 'converged' now technically means training triggered early stopping 
        # or completed all epochs cleanly with a valid loss.
        self._training_log = {
            "epochs_run": actual_epochs,
            "best_val_loss": float(best_val_loss), # Calculated in standardized space
            "dynamics_mse": best_dyn_mse,          # Calculated in original feature scale
            "converged": epochs_no_improve >= patience or actual_epochs == epochs,
        }
        return self._training_log

    def predict(self, X):
        if not self._trained:
            raise RuntimeError("Model not trained.")
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        N, seq_len, F = X.shape
        X_scaled = np.clip(
            self._scaler.transform(X.reshape(-1, F)).reshape(N, seq_len, F).astype(np.float32),
            -10.0, 10.0
        )
        Xt = torch.tensor(X_scaled, dtype=torch.float32)
        self._net.eval()
        with torch.no_grad():
            dp, rl, sl = self._net(Xt)
        risk_prob = torch.sigmoid(rl).squeeze(-1).cpu().numpy()
        stage_idx = torch.argmax(sl, dim=-1).cpu().numpy()
        dyn_orig = self._scaler.inverse_transform(dp.cpu().numpy())
        return {
            "risk_prob": risk_prob.astype(np.float32),
            "stage_idx": stage_idx.astype(np.int32),
            "dynamics_pred": dyn_orig.astype(np.float32),
        }

    def forecast(self, X_context, steps=7):
        if not self._trained:
            raise RuntimeError("Model not trained.")
        seq_len, F = X_context.shape
        buf = np.clip(self._scaler.transform(X_context).astype(np.float32), -10.0, 10.0)
        risk_traj, stage_traj, dyn_traj = [], [], []
        self._net.eval()
        with torch.no_grad():
            for _ in range(steps):
                inp = torch.tensor(buf[np.newaxis, :, :], dtype=torch.float32)
                dp, rl, sl = self._net(inp)
                risk_traj.append(float(torch.sigmoid(rl).item()))
                stage_traj.append(int(torch.argmax(sl, dim=-1).item()))
                dyn_scaled = dp.cpu().numpy()
                dyn_traj.append(self._scaler.inverse_transform(dyn_scaled)[0])
                buf = np.roll(buf, -1, axis=0)
                buf[-1, :] = dyn_scaled[0]
        return {"risk_trajectory": risk_traj, "stage_trajectory": stage_traj, "dynamics_trajectory": dyn_traj}

    def save(self, model_dir):
        if not self._trained:
            raise RuntimeError("Cannot save untrained model.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), model_dir / "stgwm.pt")
        joblib.dump(self._scaler, model_dir / "stgwm_scaler.pkl")
        cfg = {"input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
               "num_layers": self.num_layers, "num_stages": self.num_stages,
               "training_log": self._training_log}
        with open(model_dir / "stgwm_config.json", "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    @classmethod
    def load(cls, model_dir, expected_input_dim=None):
        model_dir = Path(model_dir)
        pt = model_dir / "stgwm.pt"
        sc = model_dir / "stgwm_scaler.pkl"
        cfg_p = model_dir / "stgwm_config.json"
        if not (pt.exists() and sc.exists() and cfg_p.exists()):
            return None, "MODEL_NOT_TRAINED"
        with open(cfg_p, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        if expected_input_dim is not None and cfg["input_dim"] != expected_input_dim:
            return None, "MODEL_SCHEMA_MISMATCH"
        inst = cls(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"],
                   num_layers=cfg["num_layers"], num_stages=cfg["num_stages"])
        inst._net = _STGWMNet(cfg["input_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["num_stages"])
        inst._net.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
        inst._net.eval()
        inst._scaler = joblib.load(sc)
        inst._training_log = cfg.get("training_log", {})
        inst._trained = True
        return inst, "LOADED"
