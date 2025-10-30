import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import joblib
import os

import config as cfg
import structured_processor as dp
from structured_extractor import StructuredFeatureExtractor
from dataset import RADataset
from model import CHAN_Model
import utils
import config as cfg

# --- 1. 配置和加载工件 ---

# 假设我们使用 Fold 1 的模型进行评估。
# 如果想更换，请修改此变量。
FOLD_TO_USE = cfg.FOLD_TO_USE
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, f"best_model_fold_{FOLD_TO_USE}.pth")
EXTRACTOR_PATH = os.path.join(MODEL_DIR, f"extractor_fold_{FOLD_TO_USE}.pkl")


def load_prediction_artifacts(model_path, extractor_path, device=cfg.DEVICE):
    """
    加载保存的模型和特征提取器。
    """
    if not os.path.exists(extractor_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"缺少 Fold {FOLD_TO_USE} 的模型或提取器文件。请检查路径:\n"
            f"模型: {model_path}\n提取器: {extractor_path}"
        )

    print(f"正在加载特征提取器: {extractor_path}")
    extractor = joblib.load(extractor_path)

    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_config = checkpoint.get('model_config')
    ds = model_config['ds']
    du = model_config['du']

    # 初始化模型
    model = CHAN_Model(ds=ds, du=du).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设为评估模式

    print(f"Fold {FOLD_TO_USE} 模型和提取器加载成功。")
    return model, extractor


# --- 2. 核心评估函数 ---

def predict_and_evaluate(model, dataloader, device):
    """
    使用加载的模型评估 DataLoader 中的所有数据。
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_ps = []
    all_pu = []

    with torch.no_grad():
        for hs, hu, ps, pu, labels in dataloader:
            hs, hu = hs.to(device), hu.to(device)
            ps, pu = ps.to(device), pu.to(device)
            labels = labels.to(device)

            # 我们只需要模型输出的第一个值 (p_pred)
            p_pred, _, _, _, _ = model(hs, hu, ps, pu)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(p_pred.cpu().numpy())
            all_ps.append(ps.cpu().numpy())
            all_pu.append(pu.cpu().numpy())

    # 展平所有结果
    all_labels = np.concatenate(all_labels).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_ps = np.concatenate(all_ps).flatten()
    all_pu = np.concatenate(all_pu).flatten()

    # 计算所有指标
    metrics = utils.calculate_metrics(all_labels, (all_preds > 0.5), all_preds)
    contra_acc = utils.calculate_contradictory_accuracy(all_labels, all_ps, all_pu, all_preds)

    return metrics, contra_acc


def main():
    print(f"--- 使用 Fold {FOLD_TO_USE} 模型评估完整数据集 ---")

    try:
        # 1. 加载工件
        model, extractor = load_prediction_artifacts(MODEL_PATH, EXTRACTOR_PATH, cfg.DEVICE)

        # 2. 加载和预处理所有数据
        print("\n步骤 1: 加载和预处理所有数据...")
        df_full = dp.preprocess_main(
            cfg.STRUCTURED_TEST_CSV,
            cfg.UNSTRUCTURED_TEST_CSV,
            cfg.LABEL_COLUMN,
            cfg.LAB_FEATURES,
            cfg.LAB_FEATURES
        )

        if df_full is None:
            print("数据加载或预处理失败，程序终止。")
            return

        # 提取数据
        X_structured_all = df_full[cfg.STRUCTURED_FEATURES]
        y_all = df_full[cfg.LABEL_COLUMN]
        hu_cols = list(range(cfg.DU))
        pu_col = cfg.DU
        X_unstructured_hu_all = df_full[hu_cols].values
        unstructured_pu_all = df_full[pu_col].values

        # 3. 转换所有结构化数据
        print("步骤 2: 转换结构化数据...")
        hs_all, ps_all, _ = extractor.transform(X_structured_all)

        # 4. 创建 DataLoader
        print("步骤 3: 准备 DataLoader...")
        full_dataset = RADataset(hs_all, X_unstructured_hu_all, ps_all, unstructured_pu_all, y_all.values)
        full_loader = DataLoader(full_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

        # 5. 运行预测和评估
        print("步骤 4: 运行预测和指标计算...")
        metrics, contra_acc = predict_and_evaluate(model, full_loader, cfg.DEVICE)
        acc, prec, rec, f1 = metrics

        # 6. 输出结果
        print("\n--- 完整数据集预测结果 (基于 Fold {} 模型) ---".format(FOLD_TO_USE))
        print(f"  Accuracy (准确率): {acc:.4f}")
        print(f"  Precision (精确率): {prec:.4f}")
        print(f"  Recall (召回率): {rec:.4f}")
        print(f"  F1-Score (F1 分值): {f1:.4f}")
        print(f"  Contradictory Accuracy (矛盾样本准确率): {contra_acc:.4f}")


    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("请确保你已成功运行 train.py 并生成了指定的模型文件。")
    except Exception as e:
        print(f"\n发生意外错误: {e}")


if __name__ == "__main__":
    main()