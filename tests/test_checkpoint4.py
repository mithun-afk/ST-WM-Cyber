import pytest
import numpy as np
from src2.models.evaluate import evaluate_model

class DummyPredictModel:
    def __init__(self, probs, stages, status='TRAINED'):
        self.probs = np.array(probs)
        self.stages = np.array(stages)
        self.status = status
        
    def predict(self, X):
        return {
            'risk_prob': self.probs,
            'stage_idx': self.stages
        }

def test_evaluate_model_basic():
    y_gt = np.array([0, 0, 1, 1])
    y_stage_gt = np.array([0, 0, 3, 3])
    
    # Perfect predictions
    model = DummyPredictModel(probs=[0.1, 0.2, 0.9, 0.8], stages=[0, 0, 3, 3])
    
    # Dummy X
    X = np.zeros((4, 5, 25))
    
    metrics = evaluate_model(model, X, y_gt, y_stage_gt, threshold=0.5)
    
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['fpr'] == 0.0
    assert metrics['stage_accuracy'] == 1.0
    assert metrics['confusion_matrix']['tp'] == 2
    assert metrics['confusion_matrix']['tn'] == 2

def test_evaluate_model_fpr():
    y_gt = np.array([0, 0, 0, 0])
    # One false positive
    model = DummyPredictModel(probs=[0.1, 0.9, 0.1, 0.1], stages=[0, 0, 0, 0])
    X = np.zeros((4, 5, 25))
    
    metrics = evaluate_model(model, X, y_gt, None, threshold=0.5)
    
    assert metrics['f1'] == 0.0
    assert metrics['fpr'] == 0.25 # 1 FP out of 4 negatives
    assert metrics['confusion_matrix']['fp'] == 1
    assert metrics['confusion_matrix']['tn'] == 3

def test_evaluate_model_not_trained():
    model = DummyPredictModel(probs=[], stages=[], status='NOT_TRAINED')
    X = np.zeros((1, 5, 25))
    y_gt = np.array([0])
    
    with pytest.raises(ValueError):
        evaluate_model(model, X, y_gt)
