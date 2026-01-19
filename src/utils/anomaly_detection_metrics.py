import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve

from src.utils.visualizations import plot_rov_curve



def tpr_at_fixed_fpr(y_true, y_score, fpr_level=0.01):
    """
    How effective is the method at a low false positive rate (false alarm rate)?
    If the methods produces too many false alarms it is not useful.

    Return TPR (recall) at a fixed FPR level.
    y_true : int array, 1 = anomaly, 0 = normal
    y_score : anomaly score (higher = more anomalous)
    fpr_level : scalar in [0,1]  (0.01  → 1 %)
    """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    # interpolate TPR at the requested FPR
    if fpr_level >= fpr[-1]:                # corner case: FPR too high
        return tpr[-1]
    return np.interp(fpr_level, fpr, tpr)


def fpr_at_95_tpr(y_true, y_score):
    """
    FPR when TPR = 0.95  (= 95 % anomalies recalled).
    """
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    if 0.95 >= tpr[-1]:               # 95 % outside observed range
        return fpr[-1]
    return np.interp(0.95, tpr, fpr)  # linear interp




def calculate_classification_metrics(y_pred, y_true, y_score=None, ad_name="", output_dir="", calc_acc_per_class=True):
    assert len(y_pred) == len(y_true), "Must be of same length"
    res_dict = {}
    # Accuracy  
    res_dict['accuracy_overall'] = accuracy_score(y_pred=y_pred, y_true=y_true)
    # F1 Score
    res_dict['f1_score'] = f1_score(y_pred=y_pred, y_true=y_true, zero_division=0.0)
    if y_score is not None:
        # AUROC score
        res_dict['roc_auc_score'] = float(roc_auc_score(y_score=y_score, y_true=y_true))
        
        res_dict['fpr_95tpr']    = fpr_at_95_tpr(y_true=y_true, y_score=y_score)
        res_dict['tpr_at_fpr=1'] = tpr_at_fixed_fpr(y_true=y_true, y_score=y_score, fpr_level=0.01)
        res_dict['tpr_at_fpr=5'] = tpr_at_fixed_fpr(y_true=y_true, y_score=y_score, fpr_level=0.05)
        plot_rov_curve(y_true     = y_true, 
                       scores     = y_score, 
                       output_dir = output_dir, 
                       ad_method  = ad_name)

    else:
        res_dict['roc_auc_score'] = "not available"
        res_dict['fpr_95tpr']     = "not available"
        res_dict['tpr_at_fpr=1']  = "not available"
        res_dict['tpr_at_fpr=5']  = "not available"

    # Matthews Correlation Coefficient
    res_dict['mcc'] = float(matthews_corrcoef(y_pred=y_pred, y_true=y_true))

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_pred=y_pred, y_true=y_true, labels=[0, 1]).ravel().tolist()
    res_dict['true_positives']  = int(tp)
    res_dict['false_positives'] = int(fp)
    res_dict['false_negatives'] = int(fn)
    res_dict['true_negatives']  = int(tn)
    # False Alarm Rate
    res_dict['fpr'] = float(fp / (fp +tn))
    # True positive rate / True detection rate:   TPR = TP / (TP + FN)
    res_dict['tpr'] = float(recall_score(y_true=y_true, y_pred=y_pred, pos_label=1, zero_division=0.0))


    res_dict['ratio'] = {'y_pred': {str(k): int(v) for k,v in   zip(np.unique(y_pred, return_counts=True)[0],  np.unique(y_pred, return_counts=True)[1])},
                         'y_true': {str(k): int(v) for k,v in   zip(np.unique(y_true, return_counts=True)[0],  np.unique(y_true, return_counts=True)[1])}}
    return res_dict
