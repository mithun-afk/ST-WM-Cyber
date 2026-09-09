import numpy as np
import torch

def compute_gradient_saliency(model, X_tensor):
    """
    Computes |d(risk_logit)/d(X)| averaged over sequence dimension.
    model: STGWMModel
    X_tensor: torch.Tensor shape (1, seq_len, input_dim) (Already scaled)
    """
    model._net.eval()
    
    # We expect X_tensor to be pre-scaled, but we need it as a tensor requiring grad
    X_req = X_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass - unpack 3-tuple
    dyn, risk_logit, stage = model._net(X_req)
    
    # Backward pass
    risk_logit.sum().backward()
    
    # Saliency score: mean absolute gradient over the sequence dimension
    saliency = X_req.grad.abs().squeeze(0).mean(dim=0).numpy()
    return saliency

def get_important_features(model, X, feature_cols, top_n=10):
    """
    Returns ranked important features based on gradient saliency computed on the mean window.
    X: np.array shape (N, seq_len, F) (Unscaled)
    """
    # Use the mean window across the batch to calculate global saliency
    X_mean = np.mean(X, axis=0, keepdims=True)
    
    # Transform using model's scaler
    if getattr(model, 'scaler', None) is not None:
        # Reshape to 2D for scaler, then back to 3D
        seq_len, num_features = X_mean.shape[1], X_mean.shape[2]
        X_flat = X_mean.reshape(-1, num_features)
        X_scaled = model.scaler.transform(X_flat).reshape(1, seq_len, num_features)
    else:
        X_scaled = X_mean
        
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    saliency = compute_gradient_saliency(model, X_tensor)
    
    # Normalize 0-1
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency_norm = (saliency - s_min) / (s_max - s_min)
    else:
        saliency_norm = np.zeros_like(saliency)
        
    # Rank features
    ranked_indices = np.argsort(saliency_norm)[::-1]
    
    important_features = []
    for rank, idx in enumerate(ranked_indices[:top_n]):
        important_features.append({
            'feature': feature_cols[idx],
            'importance': float(saliency_norm[idx]),
            'rank': rank + 1
        })
        
    return important_features

def get_important_windows(per_window_risk, timestamps=None, threshold=0.5):
    """
    Returns list of flagged windows based on risk threshold.
    """
    flagged = []
    for idx, risk in enumerate(per_window_risk):
        if risk > threshold:
            severity = 'HIGH' if risk > 0.8 else 'MEDIUM'
            flagged.append({
                'window_idx': int(idx),
                'risk': float(risk),
                'timestamp': timestamps[idx] if timestamps is not None and idx < len(timestamps) else None,
                'severity': severity
            })
    return flagged
