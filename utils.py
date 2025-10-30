# utils.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def calculate_metrics(y_true, y_pred, y_prob):
    """计算分类指标 """
    # y_pred 是概率，转换为 0/1
    y_pred_binary = (y_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    return accuracy, precision, recall, f1


def calculate_contradictory_accuracy(y_true, ps, pu, y_pred):
    """计算矛盾样本准确率 """
    ps_pred = (ps > 0.5).astype(int)
    pu_pred = (pu > 0.5).astype(int)

    # 找到矛盾样本的索引
    contradictory_mask = (ps_pred != pu_pred)

    if np.sum(contradictory_mask) == 0:
        return 0.0  # 没有矛盾样本

    contra_true = y_true[contradictory_mask]
    contra_pred = y_pred[contradictory_mask]

    if len(contra_true) == 0:
        return 0.0

    contra_acc = accuracy_score(contra_true, (contra_pred > 0.5).astype(int))
    return contra_acc


def plot_tsne(features, labels, title, save_path):
    """绘制 t-SNE 可视化 """
    print(f"正在生成 t-SNE: {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=['Non-RA', 'RA'])
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(save_path)
    print(f"t-SNE 图像已保存至: {save_path}")