import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss, accuracy_score

def tune_threshold(model, X_val, y_risk_val):
    """
    Finds the optimal risk threshold that maximizes F1 score on the validation set.
    """
    if model.status != 'TRAINED':
        raise ValueError("Model must be trained before tuning.")
        
    preds = model.predict(X_val)
    risk_probs = preds['risk_prob']
    
    best_f1 = -1
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_bin = (risk_probs >= thresh).astype(int)
        f1 = f1_score(y_risk_val, y_pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def evaluate_model(model, X, y_risk_gt, y_stage_gt=None, y_dyn_gt=None, threshold=0.5):
    """
    Evaluates the model on test data and returns a comprehensive metrics dictionary.
    """
    if model.status != 'TRAINED':
        raise ValueError("Model must be trained before evaluation.")
        
    preds = model.predict(X)
    risk_probs = preds['risk_prob']
    stage_preds = preds['stage_idx']
    
    # Binary metrics
    y_pred_bin = (risk_probs >= threshold).astype(int)
    
    # Handle case where all ground truths might be 0 (e.g., benign-only folds) or 1
    if len(np.unique(y_risk_gt)) > 1:
        precision = precision_score(y_risk_gt, y_pred_bin, zero_division=0)
        recall = recall_score(y_risk_gt, y_pred_bin, zero_division=0)
        f1 = f1_score(y_risk_gt, y_pred_bin, zero_division=0)
        brier = brier_score_loss(y_risk_gt, risk_probs)
    else:
        if y_risk_gt[0] == 0:
            # If only benign traffic, precision/recall/f1 are 0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            # If only attack traffic
            precision = precision_score(y_risk_gt, y_pred_bin, zero_division=0)
            recall = recall_score(y_risk_gt, y_pred_bin, zero_division=0)
            f1 = f1_score(y_risk_gt, y_pred_bin, zero_division=0)
            
        brier = np.mean((risk_probs - y_risk_gt)**2) # manual brier for single class
        
    cm = confusion_matrix(y_risk_gt, y_pred_bin)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Single class cm
        if y_risk_gt[0] == 0:
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0,0]
            
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Stage accuracy
    stage_acc = None
    if y_stage_gt is not None:
        stage_acc = accuracy_score(y_stage_gt, stage_preds)
        
    # ------------------------------------------------------------------
    # P2: Evidence Metrics
    # ------------------------------------------------------------------
    dyn_mse = None
    multi_horizon_mse = None
    early_warning_steps = 0.0

    if y_dyn_gt is not None and 'dynamics_pred' in preds:
        # True future-state benchmark (1-step S(t+1))
        # Note: y_dyn_gt is expected to be unscaled (or we compare in scaled space).
        # We compare in raw space if both are unscaled, but dynamics_pred is unscaled output.
        # Wait, dynamics_pred from inference might be raw or scaled? 
        # model.predict returns unscaled dynamics_pred.
        dyn_mse = float(np.mean((preds['dynamics_pred'] - y_dyn_gt) ** 2))
        
        # Multi-horizon forecast & Early-warning metric
        # We evaluate on a subset to keep it fast
        H = 5 # 5 steps ahead horizon
        n_evals = min(len(X) - H, 200)
        
        if n_evals > 0:
            step_mses = []
            ew_lead_times = []
            
            # Subsample to speed up
            indices = np.linspace(0, len(X) - H - 1, n_evals, dtype=int)
            for idx in indices:
                fc = model.forecast(X[idx], steps=H)
                
                # Multi-horizon MSE against ground truth trajectories
                true_traj = y_dyn_gt[idx : idx + H]
                pred_traj = np.array(fc['dynamics_trajectory'])
                step_mses.append(np.mean((pred_traj - true_traj) ** 2))
                
                # Early warning lead time
                # Look at the ground truth risk for the next H steps
                true_future_risk = y_risk_gt[idx + 1 : idx + 1 + H]
                pred_future_risk = fc['risk_trajectory']
                
                # If there's an attack in the horizon that hasn't started yet at idx
                if y_risk_gt[idx] == 0 and sum(true_future_risk) > 0:
                    attack_step = np.argmax(true_future_risk) # 0-indexed step where attack hits
                    
                    # Did the model forecast an attack BEFORE the actual attack step?
                    for s in range(attack_step + 1):
                        if pred_future_risk[s] >= threshold:
                            # Flagged at step s, attack happens at attack_step
                            # Lead time is (attack_step - s) + 1 windows
                            ew_lead_times.append((attack_step - s) + 1)
                            break
                            
            if step_mses:
                multi_horizon_mse = float(np.mean(step_mses))
            if ew_lead_times:
                early_warning_steps = float(np.mean(ew_lead_times))

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'brier_score': brier,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'stage_accuracy': stage_acc,
        'threshold_used': float(threshold),
        'dynamics_mse_1step': dyn_mse,
        'dynamics_mse_multi': multi_horizon_mse,
        'early_warning_lead_windows': early_warning_steps
    }
