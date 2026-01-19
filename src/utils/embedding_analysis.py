import os
import numpy as np
import matplotlib.pyplot as plt  

from scipy.stats import ks_2samp

from src.utils.distance_metrics import cosine_distance, cosine_similarity


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += (p.grad.detach() ** 2).sum().item()
    return np.sqrt(total)

def safe_list(x):
    return np.array(x).tolist()

def get_stats(X):
    return {
        'mean': X.mean(),
        'std': X.std(),
        'norm_mean': np.linalg.norm(X, axis=1).mean(),
        'norm_std': np.linalg.norm(X, axis=1).std()
    }

def cohens_d(a,b):
    na, nb = len(a), len(b)
    sa = a.std(ddof=1); sb = b.std(ddof=1)
    s = np.sqrt(((na-1)*sa*sa + (nb-1)*sb*sb) / (na+nb-2))
    return (a.mean() - b.mean())/s

def eig_summary(cov):
    # returns eigenvalue-based summaries (sorted descending)
    eigvals = np.linalg.eigvalsh(cov)  # ascending
    eigvals = np.maximum(eigvals, 0.0)
    eigvals_sorted = np.sort(eigvals)[::-1]  # descending
    total = eigvals_sorted.sum()
    explained_ratio = (eigvals_sorted / (total + 1e-12)).tolist()
    return {
        "eigenvalues": safe_list(eigvals_sorted.tolist()),
        "explained_ratio": explained_ratio,
        "sum": float(total),
        "max": float(eigvals_sorted[0]) if len(eigvals_sorted) else 0.0,
        "min": float(eigvals_sorted[-1]) if len(eigvals_sorted) else 0.0,
        "condition_number": float((eigvals_sorted[0] / (eigvals_sorted[-1] + 1e-12)) if len(eigvals_sorted) > 0 else 0.0),
        "desc_eigenvalues":     'Eigenvalues should not be close to 0 (e.g., < 1e-6) -> if only 1-2 are significantly > 0 -> space is collapsed',
        "desc_explained_ratio": 'Good: spread out, e.g. top 5 eigenvalues explain ca. 60%. of variance, not all 99%. Bad: top 1 eigenvalue explains ca. 90 - 100 percent -> effectively 1-D embeddings.',
    }


def obtain_statistics(train_z_future_np, train_z_future_pred_np, test_z_future_np, test_z_future_pred_np, train_z, test_z, test_label, output_dir):

    ### Cosine Distance
    train_cosine_distance = cosine_distance(train_z_future_np, train_z_future_pred_np, diagonal_only=True)
    test_cosine_distance  = cosine_distance(test_z_future_np, test_z_future_pred_np, diagonal_only=True)
    cosine_distances = {
                    'train_mean':   float(np.mean(train_cosine_distance)),
                    'train_all':    train_cosine_distance.tolist(),
                    'test_mean':    float(np.mean(test_cosine_distance)),
                    'test_all':     test_cosine_distance.tolist(),
                    'test_label':   test_label,
                    'description':  'Range = [0, 2], with 0 = identical vectors, 1 = orthogonal and 2 = completely opposite'
                }
    if True:    
        plt.hist(train_cosine_distance, bins=100, alpha=0.6, label='train')
        plt.hist(test_cosine_distance,  bins=100, alpha=0.6, label='test')
        plt.legend(); 
        plt.title('cosine distance distrib'); 
        plt.savefig(os.path.join(output_dir, 'cosine_distance_distrib.png'))
        plt.close()

    ### Cosine Similarity
    train_cosine_similarities = cosine_similarity(train_z_future_np, train_z_future_pred_np, diagonal_only=True)
    test_cosine_similarities  = cosine_similarity(test_z_future_np, test_z_future_pred_np, diagonal_only=True)
    cosine_similarities = {
                'train_mean':   float(np.mean(train_cosine_similarities)),
                'train_all':    train_cosine_similarities.tolist(),
                'test_mean':    float(np.mean(test_cosine_similarities)),
                'test_all':     test_cosine_similarities.tolist(),
                'test_label':   test_label,
                'description':  'Range = [-1, 1], with -1 = opposite direction, 0 = orthogonal and 1 = identical'
            }
    if True:              
        plt.hist(train_cosine_similarities, bins=100, alpha=0.6, label='train')
        plt.hist(test_cosine_similarities,  bins=100, alpha=0.6, label='test')
        plt.legend(); 
        plt.title('cosine similarity distrib'); 
        plt.savefig(os.path.join(output_dir, 'cosine_similarity_distrib.png'))
        plt.close()

    ### Statistics
    stats = {
                'train_z':                          get_stats(train_z),
                'train_z_future':                   get_stats(train_z_future_np),
                'train_z_future_pred':              get_stats(train_z_future_pred_np),
                'test_z':                           get_stats(test_z),
                'test_z_future':                    get_stats(test_z_future_np),
                'test_z_future_pred':               get_stats(test_z_future_pred_np),
                'test_z_diff_anomaly_only':         get_stats(test_z[np.array(test_label)==1]),
                'test_z_diff_normal_only':          get_stats(test_z[np.array(test_label)==0]),
                'test_z_future_pred_anomaly_only':  get_stats(test_z_future_pred_np[np.array(test_label)==1]),
                'test_z_future_pred_normal_only':   get_stats(test_z_future_pred_np[np.array(test_label)==0]),
            }


    ### Monitoring of the variance per embedding dimension    
    var_per_dim = {'train_var_per_dim': np.var(train_z_future_pred_np, axis=0).tolist(),
                   'test_var_per_dim': np.var(test_z_future_pred_np, axis=0).tolist(),
                   'description': 'Variance per embedding dimension for [datasplit]_z_future_pred_np'
                  }
    
    ### Embedding Corviarance: covariance summaries + eigenspectra 
    cov_train = np.cov(train_z_future_np.T) if train_z_future_np.shape[0] > 1 else np.zeros((train_z_future_np.shape[1],train_z_future_np.shape[1]))
    cov_test  = np.cov(test_z_future_np.T)  if test_z_future_np.shape[0]  > 1 else np.zeros((test_z_future_np.shape[1],test_z_future_np.shape[1]))

    cov_summary = {
        'train_cov_eig_summary': eig_summary(cov_train),
        'test_cov_eig_summary': eig_summary(cov_test),
        # add trace / rank
        'train_trace': float(np.trace(cov_train)),
        'test_trace': float(np.trace(cov_test)),
        'train_rank': int(np.linalg.matrix_rank(cov_train)),
        'test_rank': int(np.linalg.matrix_rank(cov_test)),
    }

            
    ### Statistical tst (KS-test)            
    ks_stat, pval = ks_2samp(train_z, test_z)
    ks_res =  {"KS stat": ks_stat.tolist(),
                       "pval":    pval.tolist(),
                       "input": "Difference",
                       "desc": "High p-value: distributions are not significantly different / Low p-value: good — distributions differ."
                    }

    ### Cohen's d
    cohen_d_res = {"value": cohens_d(train_z, test_z),
                           "input": "Difference",
                           "desc":  "Value indicates the effect size between two distributions. d=0 small; d>=0.5 moderate; d>=0.8 large."}


    return {'cosine_distances': cosine_distances,
            'cosine_similarities': cosine_similarities,
            'stats':        stats,
            'var_per_dim':  var_per_dim,            
            'cov_summary': cov_summary,
            'ks_test': ks_res,
            'Cohen_d': cohen_d_res,
            }