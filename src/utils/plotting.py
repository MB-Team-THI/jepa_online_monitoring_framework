import matplotlib.pyplot as plt


def plot_series_with_anomalies(series, label, title='Time Series'):
    plt.figure(figsize=(10, 4))
    for i in range(series.shape[1]):
        plt.plot(series[:, i], label=f'Feature {i+1}')
    if label:
        plt.title(f"{title} [ANOMALY]")
    else:
        plt.title(f"{title} [Normal]")
    plt.legend()
    plt.show()
