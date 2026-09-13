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
    
    lr_features = []
    # If we have the LR pipeline, use its weights for present risk explainability
    if lr_scorer is not None and hasattr(lr_scorer, 'named_steps'):
        try:
            # We use the current window X[-1] instead of the mean
            X_last = X[-1] # shape (seq_len, F)
            
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
            X_scaled_lr = scaler.transform(X_last.reshape(1, -1))[0]
            avg_scaled_input = np.zeros(F)
            for step in range(seq_len):
                avg_scaled_input += X_scaled_lr[step * F : (step + 1) * F]
            avg_scaled_input /= seq_len
            
            saliency_lr = np.abs(avg_weights * avg_scaled_input)
            
            # Normalize
            s_max = saliency_lr.max()
            if s_max > 0:
                saliency_norm_lr = saliency_lr / s_max
            else:
                saliency_norm_lr = np.zeros_like(saliency_lr)
                
            ranked_indices_lr = np.argsort(saliency_norm_lr)[::-1]
            
            for rank, idx in enumerate(ranked_indices_lr[:top_n]):
                if saliency_norm_lr[idx] > 0.001:
                    lr_features.append({
                        'feature': feature_cols[idx],
                        'importance': float(saliency_norm_lr[idx]),
                        'rank': rank + 1
                    })
        except Exception as e:
            print("LR XAI error:", e)
            pass
            
    # LSTM gradient saliency for future trajectory explainability
    lstm_features = []
    try:
        X_mean = np.mean(X, axis=0, keepdims=True)
        if getattr(model, '_scaler', None) is not None:
            seq_len, num_features = X_mean.shape[1], X_mean.shape[2]
            X_flat = X_mean.reshape(-1, num_features)
            X_scaled_lstm = model._scaler.transform(X_flat).reshape(1, seq_len, num_features)
        else:
            X_scaled_lstm = X_mean
            
        import torch
        X_tensor = torch.tensor(X_scaled_lstm, dtype=torch.float32)
        saliency_lstm = compute_gradient_saliency(model, X_tensor)
        
        s_min, s_max = saliency_lstm.min(), saliency_lstm.max()
        if s_max > s_min:
            saliency_norm_lstm = (saliency_lstm - s_min) / (s_max - s_min)
        else:
            saliency_norm_lstm = np.zeros_like(saliency_lstm)
            
        ranked_indices_lstm = np.argsort(saliency_norm_lstm)[::-1]
        for rank, idx in enumerate(ranked_indices_lstm[:top_n]):
            if saliency_norm_lstm[idx] > 0.001:
                lstm_features.append({
                    'feature': feature_cols[idx],
                    'importance': float(saliency_norm_lstm[idx]),
                    'rank': rank + 1
                })
    except Exception as e:
        print("LSTM XAI error:", e)
        pass

    return {
        "present_risk_explanation": lr_features,
        "future_forecast_explanation": lstm_features
    }


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
