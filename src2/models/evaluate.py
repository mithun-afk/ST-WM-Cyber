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

def evaluate_model(model, X, y_risk_gt, y_stage_gt=None, threshold=0.5):
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
    
    # Handle case where all ground truths might be 0 (e.g., benign-only folds)
    if len(np.unique(y_risk_gt)) > 1:
        precision = precision_score(y_risk_gt, y_pred_bin, zero_division=0)
        recall = recall_score(y_risk_gt, y_pred_bin, zero_division=0)
        f1 = f1_score(y_risk_gt, y_pred_bin, zero_division=0)
        brier = brier_score_loss(y_risk_gt, risk_probs)
    else:
        # If only benign traffic, precision/recall/f1 are 0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
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
        
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'brier_score': brier,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'stage_accuracy': stage_acc,
        'threshold_used': float(threshold)
    }
