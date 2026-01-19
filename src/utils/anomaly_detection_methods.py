'''
Various anomaly detection methods
'''
import numpy as np

from pyod.models.abod import ABOD
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor


from src.utils.anomaly_detection_metrics import calculate_classification_metrics



def perform_anomaly_detection(x_train, x_test, y_test, settings={}, framework_task="detection_of_injected_anomalies"):
    output_dir = settings['output_dir']
    if framework_task== "detection_of_injected_anomalies":
        # Same number of anomalies and normal data samples is expected
        balanced_data = True
    else:
        balanced_data = False
    res_dict = {}

    ### LOF
    if settings['lof']['enabled']:
        res_dict['lof'] = method_lof(x_train       = x_train, 
                                     x_test        = x_test, 
                                     y_test        = y_test,
                                     settings      = settings['lof'], 
                                     balanced_data = balanced_data,
                                     output_dir    = output_dir)

    ### ABOD
    if settings['abod']['enabled']:
        res_dict['abod'] = method_abod(x_train       = x_train, 
                                       x_test        = x_test, 
                                       y_test        = y_test,
                                       settings      = settings['abod'], 
                                       balanced_data = balanced_data,
                                       output_dir    = output_dir)
    ### GMM
    if settings['gmm']['enabled']:
        res_dict['gmm'] = method_gmm(x_train       = x_train, 
                                     x_test        = x_test, 
                                     y_test        = y_test,
                                     settings      = settings['gmm'], 
                                     balanced_data = balanced_data,
                                     output_dir    = output_dir)


    return res_dict 


def method_lof(x_train, x_test, y_test, settings, return_scores=True, balanced_data=True, output_dir=""):
    # Create the model
    model_lof = LocalOutlierFactor(n_neighbors=settings['n_neighbors'], novelty=settings['novelty'])
    # Fit the model on the train set (without anomalies)
    model_lof.fit(x_train)   

    # Predict anomalies and anomaly scores for test set (50% normal and 50% anomaly)
    preds = model_lof.predict(x_test)    
    preds = [0 if x == 1 else 1 for x in preds]
    anomaly_scores_test = -1 * model_lof.score_samples(x_test)
    metrics_test = calculate_classification_metrics(y_pred=preds, y_true=y_test, y_score=anomaly_scores_test, ad_name='LOF', output_dir=output_dir)

    ad_results = {'preds_test': preds,
                  'labels_test': y_test,}
    if return_scores:
        idx_normal  = [idx for idx, i in enumerate(y_test) if i==0]
        idx_anomaly = [idx for idx, i in enumerate(y_test) if i==1]
        if balanced_data:
            assert len(idx_normal) == len(idx_anomaly), "Should be of same length"
        ad_results['ad_scores'] = {'test_split_total':      anomaly_scores_test.tolist(),
                                   'test_split_normal':     anomaly_scores_test[idx_normal].tolist(),
                                   'test_split_anomaly':    anomaly_scores_test[idx_anomaly].tolist(),
                                   } 
    else:
        ad_results['ad_scores'] = "not recorded"

    res_dict = {'metrics_test': metrics_test,
                'ad_results':   ad_results,
                'settings':     settings,
                }
    return res_dict 


def method_abod(x_train, x_test, y_test, settings, return_scores=True, balanced_data=True, output_dir=""):

    # Create and fit the train set (without anomalies)
    model_abod = ABOD().fit(x_train)

    # Define proper threshold based on val-set
    probas_test, confience_test = model_abod.predict_proba(X = np.array(x_test),
                                                         method=settings['method'], 
                                                         return_confidence=True)
    outlier_probability_test = probas_test[:,1] # ([proba of normal, proba of outliers])if 

    if [np.isnan(x) for x in outlier_probability_test].count(True) > 0:
        print("Warning: NaN values found in outlier probabilities")
        outlier_probability_test = None
    y_pred_test = model_abod.predict(X=x_test)
    metrics_test = calculate_classification_metrics(y_pred=y_pred_test, y_true=y_test, y_score=outlier_probability_test, ad_name='ABOD', output_dir=output_dir)

    ad_results = {'preds_test':  y_pred_test.tolist(),
                  'labels_test': y_test,}    
    if return_scores:
        # The anomaly score of an input sample is computed based on different detector algorithms. For consistency, outliers are assigned with larger anomaly scores.
        anomaly_scores_test = model_abod.decision_function(x_test)

        idx_normal  = [idx for idx, i in enumerate(y_test) if i==0]
        idx_anomaly = [idx for idx, i in enumerate(y_test) if i==1]
        if balanced_data:
            assert len(idx_normal) == len(idx_anomaly), "Should be of same length"
        ad_results['ad_scores'] = {'test_split_total':      anomaly_scores_test.tolist(),
                                   'test_split_normal':     anomaly_scores_test[idx_normal].tolist(),
                                   'test_split_anomaly':    anomaly_scores_test[idx_anomaly].tolist(),
                                   } 
    else:
        ad_results['ad_scores'] = "not recorded"
    res_dict = {'metrics_test': metrics_test,
                'ad_results':   ad_results,
                'settings':     settings,}
    return res_dict


def method_gmm(x_train, x_test, y_test, settings, return_scores=True, balanced_data=True, output_dir=""):
    # Define proper threshold based on train-set - based on log-likelihoods
    # Calculate log-likelihoods for training inliers and outliers
    # "Log-likelihood considers the combined contribution of all clusters rather than focusing on the most probable one."
    # "It captures the overall "fit" of the data point under the model, making it more robust for anomaly detection."
    # Negative score = the higher the number the more normal / the lower the number the more anomalous

    # Create the model
    gmm = GaussianMixture(n_components=settings['n_components'], 
                          covariance_type=settings['covariance_type'], 
                          random_state=42)
    # Fit the model on the train set (without anomalies)
    gmm.fit(x_train)
    X_train_log_likelihood_normal  = gmm.score_samples(x_train)
    percentile_val = settings['log_likelihood_threshold_percentil']
    threshold = np.percentile(X_train_log_likelihood_normal, percentile_val)

    # Predict anomalies and anomaly scores for test set (50% normal and 50% anomaly)
    X_test_log_likelihood  = gmm.score_samples(x_test)

    y_pred_test = [1 if x < threshold else 0 for x in X_test_log_likelihood]
    metrics_gmm = calculate_classification_metrics(y_pred=y_pred_test, y_true=y_test, y_score= -X_test_log_likelihood, ad_name='GMM', output_dir=output_dir)

    ad_results = {'preds_test':  y_pred_test,
                  'labels_test': y_test,}
    if return_scores:
        idx_normal  = [idx for idx, i in enumerate(y_test) if i==0]
        idx_anomaly = [idx for idx, i in enumerate(y_test) if i==1]
        if balanced_data:
            assert len(idx_normal) == len(idx_anomaly), "Should be of same length"
        ad_results['ad_scores'] = {'test_split_total':      X_test_log_likelihood.tolist(),
                                   'test_split_normal':     X_test_log_likelihood[idx_normal].tolist(),
                                   'test_split_anomaly':    X_test_log_likelihood[idx_anomaly].tolist(),
                                   } 
    else:
        ad_results['ad_scores'] = "not recorded"
    res_dict = {'metrics_test': metrics_gmm,
                'ad_results':   ad_results,
                'settings':     settings,}
    return res_dict
