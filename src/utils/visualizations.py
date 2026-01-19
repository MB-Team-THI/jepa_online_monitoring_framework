import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc


def visualize_latent_space(z, labels, output_dir="", viz_method="TSNE"):
    z_all = np.array(z)
    colors = ['blue' if label == 0 else 'red' for label in labels]

    # Dimensionality reductions
    z_pca  = PCA(n_components=2).fit_transform(z_all)
    z_tsne = TSNE(n_components=2, random_state=42).fit_transform(z_all)
    z_umap = umap.UMAP(n_components=2, random_state=42, n_jobs=1).fit_transform(z_all)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(z_pca[:, 0], z_pca[:, 1], c=colors, alpha=0.6)
    axes[0].set_title("PCA")

    axes[1].scatter(z_tsne[:, 0], z_tsne[:, 1], c=colors, alpha=0.6)
    axes[1].set_title("t-SNE")

    axes[2].scatter(z_umap[:, 0], z_umap[:, 1], c=colors, alpha=0.6)
    axes[2].set_title("UMAP")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    plt.tight_layout()

    if output_dir != "":
        # Save figure
        plot_name = 'reduced_latent_space.png'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, plot_name)
        plt.savefig(filename)
        plt.close()




def plot_ad_scores_distribution(ad_scores, output_dir, ad_method_name, plot_type="kde"):
    assert plot_type in ("kde", "hist"), "plot_type must be 'kde' or 'hist'"

    if ad_method_name == 'GMM_log_likelihood_score':
        ad_method_name = 'GMM'


    # Setup plot styles
    plt.figure(figsize=(12, 6))
    font_size_title = 35
    font_size_other = 25
    line_styles = ['-', '--', '-.', ':']
    colors = sns.color_palette("colorblind", len(ad_scores))

    for i, (setting, scores) in enumerate(ad_scores.items()):
        color = colors[i % len(colors)]
        style = line_styles[i % len(line_styles)]

        if plot_type == "kde":
            sns.kdeplot(scores, label=setting, color=color, linestyle=style, linewidth=5)
        else:  # histogram
            plt.hist(scores, bins=300, label=setting, color=color, histtype='step', linestyle=style, linewidth=5)

        # Plot mean line
        mean_score = np.mean(scores)
        plt.axvline(mean_score, color=color, linestyle=style, linewidth=4)

    plt.title(f"Anomaly Score Distribution - {ad_method_name}", fontsize=font_size_title)
    plt.xlabel("Anomaly Score", fontsize=font_size_other)
    plt.ylabel("Density" if plot_type == "kde" else "Count", fontsize=font_size_other)    
    plt.xticks(fontsize=font_size_other)
    plt.yticks(fontsize=font_size_other)
    plt.legend(title="Legend", fontsize=font_size_other)
    plt.tight_layout()
    # The limits of the x-axis must be adapted manually
    if ad_method_name == "LOF":
        plt.xlim(0.65, 3.0)
    elif ad_method_name == "ABOD":
        plt.xlim(0, 0.27)    
    elif ad_method_name == "GMM":
        plt.xlim(-200,120) 

    # Save plots
    pdf_path = os.path.join(output_dir, f"{ad_method_name}_score_distribution_{plot_type}.pdf")
    img_path = os.path.join(output_dir, f"{ad_method_name}_score_distribution_{plot_type}.png")

    plt.savefig(pdf_path)
    plt.savefig(img_path, dpi=300)
    plt.close()

def plot_ad_scores_violin_box_plot(ad_scores, output_dir, ad_method_name):
    df = pd.DataFrame({
        "Severity": np.concatenate([[k] * len(v) for k, v in ad_scores.items()]),
        "Anomaly Score": np.concatenate(list(ad_scores.values()))
    })

    # Plot

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    font_size_title = 35
    font_size_other = 25

    # Create violin plot with boxplot overlay
    sns.violinplot(
        data=df,
        y="Severity",
        x="Anomaly Score",
        inner=None,            # remove default inner bars, we’ll overlay boxplot manually
        palette="colorblind",
        linewidth=2
    )

    sns.boxplot(
        data=df,
        y="Severity",
        x="Anomaly Score",
        width=0.2,
        boxprops={"zorder": 2, "facecolor": "white", "edgecolor": "black"},
        showcaps=True,
        showfliers=False,
        whiskerprops={"linewidth": 2},
        medianprops={"color": "black", "linewidth": 2},
    )

    plt.title(f"Anomaly Scores by Severity - {ad_method_name}", fontsize=font_size_title)
    plt.xlabel("Anomaly Score", fontsize=font_size_other)
    # plt.ylabel("Severity", fontsize=font_size_other)
    plt.xticks(fontsize=font_size_other)
    plt.yticks(fontsize=font_size_other)
    plt.tight_layout()

    # Optionally adjust x-limits (as you did before)
    # if ad_method_name == "LOF":
    #     plt.xlim(0.65, 3.0)
    # elif ad_method_name == "ABOD":
    #     plt.xlim(0, 0.27)
    # elif ad_method_name == "GMM":
    #     plt.xlim(-200, 120)

    # Save plots
    pdf_path = os.path.join(output_dir, f"{ad_method_name}_violin_boxplot.pdf")
    img_path = os.path.join(output_dir, f"{ad_method_name}_violin_boxplot.png")
    plt.savefig(pdf_path)
    plt.savefig(img_path, dpi=300)
    plt.close()





def plot_rov_curve(y_true, scores, output_dir, ad_method):

    fpr, tpr, _ = roc_curve(y_true, scores)     # 1 = anomaly, 0 = normal
    roc_auc = auc(fpr, tpr)                     # identical to sklearn.metrics.roc_auc_score

    output_dir = os.path.join(output_dir, "roc_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### Log scale on x-axis
    plt.figure(figsize=(4,4))
    plt.plot(fpr, tpr, lw=2, label=f'AUROC = {roc_auc:0.3f}')
    # plt.plot([0,1], [0,1], 'k--', lw=1)        # random classifier
    plt.xscale('log')                           # ← automotive secret: log-FPR shows 0.1 % region
    plt.xlabel('False Positive Rate  (log scale)')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC curve - {ad_method}")
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    # Save plots
    pdf_path = os.path.join(output_dir, f"ROC_{ad_method}_log.pdf")
    img_path = os.path.join(output_dir, f"ROC_{ad_method}_log.png")
    plt.savefig(pdf_path)
    plt.savefig(img_path, dpi=300)
    plt.close()

    ### Normal x-axis
    plt.figure(figsize=(4, 4.))
    plt.plot(fpr, tpr, lw=2, label=f'AUROC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')       # random classifier

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC curve - {ad_method}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # Save plots
    pdf_path = os.path.join(output_dir, f"ROC_{ad_method}.pdf")
    img_path = os.path.join(output_dir, f"ROC_{ad_method}.png")
    plt.savefig(pdf_path)
    plt.savefig(img_path, dpi=300)
    plt.close()




