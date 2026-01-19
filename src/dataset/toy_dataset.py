import os
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt



class JEPA_TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, config):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
        self.seq_len_hist = config['seq_len_hist']
        self.seq_len_pred = config['seq_len_pred']

    def __getitem__(self, idx):
        x = self.data[idx]
        seq_len = x.shape[0]
        objects_dict = {'past':   x[:self.seq_len_hist],
                        'future': x[self.seq_len_hist:],
                        'whole':  x,
                        'label':  self.labels[idx],
                        }
    
        return {'objects': objects_dict}

    def __len__(self):
        return len(self.data)



def get_synthetic_dataset(n_samples=1000, with_anomalies=True, config={}):
    data = []
    labels = []
    sample_not_saved_yet = config['save_one_sample']
    anomaly_ratio        = config['anomaly_ratio']
    
    for _ in range(n_samples):
        x = np.linspace(0, 4 * np.pi, config['seq_len'])
        series = np.stack([
            np.sin(x + np.random.rand()) + np.random.normal(0, 0.1, config['seq_len']),
            np.cos(x + np.random.rand()) + np.random.normal(0, 0.1, config['seq_len']),
            np.sin(2 * x + np.random.rand()) + np.random.normal(0, 0.1, config['seq_len'])
        ], axis=1)

        if with_anomalies and (np.random.rand() < anomaly_ratio):
            time_range = np.arange(config['seq_len_hist'], config['seq_len'])
            time_idx = np.random.choice(time_range, size=config['n_times_anomaly'], replace=False)
            for t in time_idx:
                mu = config['anomaly_noise']['mu']
                sigma = config['anomaly_noise']['sigma']
                anomaly = np.random.normal(mu, sigma, size=config['input_dim'])
                series[t] += anomaly
            labels.append(1)

            if sample_not_saved_yet:
                fig, axes = plt.subplots(1, 1, figsize=(18, 5))
                axes.plot(series[:, 0], alpha=0.6)
                axes.plot(series[:, 1], alpha=0.6)
                axes.plot(series[:, 2], alpha=0.6)
                plt.scatter(x=time_idx, y=[0 for x in time_idx], color='red', marker='x')
                axes.axvline(x = config['seq_len_hist']-0.5, color = 'b', label = 'end history', linestyle='--')
                plt.xlim(-0.5, config['seq_len']-0.5)
                axes.set_title("Anomalies within the time-series data")
                sample_not_saved_yet = False


                plot_name = 'sample_anomaly_in_time_series_data.png'
                output_dir = config['output_dir']
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                filename = os.path.join(output_dir, plot_name)
                plt.savefig(filename)
                plt.close()

        else:
            labels.append(0)
        data.append(series)

    if not with_anomalies:
        assert 1 not in labels, "there should be no anomaly here"

    return np.array(data), np.array(labels)