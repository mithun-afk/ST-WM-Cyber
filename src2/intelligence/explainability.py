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

def get_important_features(model, lr_scorer, X, feature_cols, top_n=10):
    import numpy as np
    
    # If we have the LR pipeline, use its weights for explainability
    if lr_scorer is not None and hasattr(lr_scorer, 'named_steps'):
        try:
            # We use the current window X[-1] instead of the mean
            X_last = X[-1] # shape (seq_len, F)
            X_last_mean = np.mean(X_last, axis=0) # shape (F,)
            
            lr = lr_scorer.named_steps["lr"]
            scaler = lr_scorer.named_steps["scaler"]
            
            # For each step in the sequence, the LR has F weights.
            # We average the F weights across the sequence length.
            weights = lr.coef_[0]
            seq_len = len(X_last)
            F = len(feature_cols)
            
            avg_weights = np.zeros(F)
            for step in range(seq_len):
                avg_weights += weights[step * F : (step + 1) * F]
            avg_weights /= seq_len
            
            # The saliency is |weight * scaled_input|
            X_scaled = scaler.transform(X_last.reshape(1, -1))[0]
            avg_scaled_input = np.zeros(F)
            for step in range(seq_len):
                avg_scaled_input += X_scaled[step * F : (step + 1) * F]
            avg_scaled_input /= seq_len
            
            saliency = np.abs(avg_weights * avg_scaled_input)
            
            # Normalize
            s_max = saliency.max()
            if s_max > 0:
                saliency_norm = saliency / s_max
            else:
                saliency_norm = np.zeros_like(saliency)
                
            ranked_indices = np.argsort(saliency_norm)[::-1]
            
            important_features = []
            for rank, idx in enumerate(ranked_indices[:top_n]):
                if saliency_norm[idx] > 0.001:
                    important_features.append({
                        'feature': feature_cols[idx],
                        'importance': float(saliency_norm[idx]),
                        'rank': rank + 1
                    })
            return important_features
        except Exception as e:
            print("LR XAI error:", e)
            pass
            
    # Fallback to LSTM gradient saliency
    X_mean = np.mean(X, axis=0, keepdims=True)
    if getattr(model, 'scaler', None) is not None:
        seq_len, num_features = X_mean.shape[1], X_mean.shape[2]
        X_flat = X_mean.reshape(-1, num_features)
        X_scaled = model.scaler.transform(X_flat).reshape(1, seq_len, num_features)
    else:
        X_scaled = X_mean
        
    import torch
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    saliency = compute_gradient_saliency(model, X_tensor)
    
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency_norm = (saliency - s_min) / (s_max - s_min)
    else:
        saliency_norm = np.zeros_like(saliency)
        
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
