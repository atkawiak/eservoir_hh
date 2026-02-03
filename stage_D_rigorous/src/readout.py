
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, balanced_accuracy_score, roc_auc_score
from typing import Dict, Any, List, Optional
from cv import BlockedCV

class ReadoutModule:
    """
    Journal-Grade Readout Module.
    Features strict Nested Blocked CV for alpha selection and leakage-proof evaluation.
    """
    def __init__(self, rng: np.random.Generator, cv_folds: int = 5, cv_gap: int = 10):
        self.rng = rng
        self.folds = cv_folds
        self.gap = cv_gap

    def train_ridge_cv(self, X: np.ndarray, y: np.ndarray, 
                       task_type: str = "regression", 
                       alphas: List[float] = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]) -> Dict[str, float]:
        """
        Trains Ridge using Nested Blocked CV.
        
        Outer Loop: Evaluate Performance (Test).
        Inner Loop: Select Hyperparameters (Validation on Train).
        """
        outer_cv = BlockedCV(self.folds, self.gap)
        
        scores = []
        best_alphas = []
        
        for train_idx, test_idx in outer_cv.split(len(y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # --- INNER CV (Hyperparameter Selection) ---
            # Strictly matches outer protocol (Blocked + Gap) but on Train data
            inner_best_alpha = self._select_alpha_nested(X_train, y_train, alphas)
            best_alphas.append(inner_best_alpha)
            
            # --- FINAL TRAIN (On Data available in this Fold) ---
            model = Ridge(alpha=inner_best_alpha)
            model.fit(X_train, y_train)
            
            # --- EVALUATE ---
            if task_type == "regression":
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                var = np.var(y_test)
                nrmse = np.sqrt(mse / (var + 1e-12))
                scores.append({'nrmse': nrmse, 'mse': mse})
                
            elif task_type == "classification":
                # Ridge for Classification (targets {0, 1})
                y_pred_score = model.predict(X_test) # Continuous score s = Xw
                y_pred_class = (y_pred_score > 0.5).astype(int)
                
                acc = accuracy_score(y_test, y_pred_class)
                bal_acc = balanced_accuracy_score(y_test, y_pred_class)
                try:
                    auc = roc_auc_score(y_test, y_pred_score)
                except ValueError:
                    auc = 0.5 # Handle single-class edge case
                    
                scores.append({'acc': acc, 'bal_acc': bal_acc, 'auc': auc})
        
        # Aggregate Results
        agg = {}
        if not scores: return {}
        
        metric_keys = scores[0].keys()
        for k in metric_keys:
            agg[k] = np.mean([s[k] for s in scores])
            
        # Log-mean of alphas for better interpretation
        mean_log_alpha = np.mean(np.log10(best_alphas))
        agg['best_alpha_mean'] = np.power(10, mean_log_alpha)
        # Also store distribution if possible? No, strict schema.
        
        return agg

    def _select_alpha_nested(self, X: np.ndarray, y: np.ndarray, alphas: List[float]) -> float:
        """
        Selects best alpha using Inner Blocked CV (3 folds) optimizing MSE.
        Optimization metric is always MSE for Ridge consistency.
        """
        if len(alphas) == 1: 
            return alphas[0]
            
        if len(X) < 20: # Fallback for very short sequences
             # Single split 50/50
             tr_len = int(len(X) * 0.5)
             tr_idx = np.arange(tr_len)
             val_idx = np.arange(tr_len, len(X))
             # Just one loop manually
             best = alphas[0]; min_loss = float('inf')
             for a in alphas:
                 m = Ridge(alpha=a); m.fit(X[tr_idx], y[tr_idx])
                 loss = mean_squared_error(y[val_idx], m.predict(X[val_idx]))
                 if loss < min_loss: min_loss = loss; best=a
             return best
             
        inner_cv = BlockedCV(n_folds=3, gap=self.gap)
        alpha_losses = {a: [] for a in alphas}
        
        for tr_idx, val_idx in inner_cv.split(len(y)):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            for a in alphas:
                m = Ridge(alpha=a)
                m.fit(X_tr, y_tr)
                pred = m.predict(X_val)
                loss = mean_squared_error(y_val, pred)
                alpha_losses[a].append(loss)
                
        # Find alpha with minimum average MSE
        best_alpha = min(alphas, key=lambda x: np.mean(alpha_losses[x]))
        return best_alpha
