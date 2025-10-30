import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib
import os

import config as cfg
import structured_processor as dp
from structured_extractor import StructuredFeatureExtractor
from dataset import RADataset
from model import CHAN_Model, CA_OCL_Loss
import utils


def train_one_epoch(model, dataloader, optimizer, criterion_ce, criterion_contrast, gamma, device):
    model.train()
    total_loss = 0.0

    for hs, hu, ps, pu, labels in dataloader:
        hs, hu = hs.to(device), hu.to(device)
        ps, pu = ps.to(device), pu.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        p_pred, Zs, Zu, proj_s, proj_u = model(hs, hu, ps, pu)

        # --- 计算总损失 ---

        # 1. 交叉熵损失 L_CE
        loss_ce = criterion_ce(p_pred, labels)

        # 2. 对比损失 L_contrast
        loss_contrast = criterion_contrast(
            Zs, Zu,
            proj_s,
            proj_u,
            ps, pu, labels
        )

        # 3. 总损失
        total_loss_batch = loss_ce + gamma * loss_contrast

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion_ce, criterion_contrast, gamma, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_ps = []
    all_pu = []

    with torch.no_grad():
        for hs, hu, ps, pu, labels in dataloader:
            hs, hu = hs.to(device), hu.to(device)
            ps, pu = ps.to(device), pu.to(device)
            labels = labels.to(device)

            p_pred, Zs, Zu, proj_s, proj_u = model(hs, hu, ps, pu)

            loss_ce = criterion_ce(p_pred, labels)
            loss_contrast = criterion_contrast(
                Zs, Zu,
                proj_s,
                proj_u,
                ps, pu, labels
            )
            total_loss_batch = loss_ce + gamma * loss_contrast
            total_loss += total_loss_batch.item()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(p_pred.cpu().numpy())
            all_ps.append(ps.cpu().numpy())
            all_pu.append(pu.cpu().numpy())

    all_labels = np.concatenate(all_labels).flatten()
    all_preds = np.concatenate(all_preds).flatten()
    all_ps = np.concatenate(all_ps).flatten()
    all_pu = np.concatenate(all_pu).flatten()

    metrics = utils.calculate_metrics(all_labels, (all_preds > 0.5), all_preds)
    contra_acc = utils.calculate_contradictory_accuracy(all_labels, all_ps, all_pu, all_preds)

    return total_loss / len(dataloader), metrics, contra_acc


def save_model(model, extractor, fold, model_dir="saved_models"):
    """保存模型和特征提取器"""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 保存模型
    model_path = os.path.join(model_dir, f"best_model_fold_{fold + 1}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'ds': model.ds,
            'du': model.du
        }
    }, model_path)

    # 保存特征提取器
    extractor_path = os.path.join(model_dir, f"extractor_fold_{fold + 1}.pkl")
    joblib.dump(extractor, extractor_path)

    print(f"模型和提取器已保存: {model_path}, {extractor_path}")


def main():
    print("--- 开始 RA 诊断项目 ---")

    # 创建模型保存目录
    model_dir = cfg.MODEL_DIR
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. 数据预处理
    print("步骤 1: 加载和预处理数据...")
    df_full = dp.preprocess_main(
        cfg.STRUCTURED_DATA_CSV,
        cfg.UNSTRUCTURED_FEAT_CSV,
        cfg.LABEL_COLUMN,
        cfg.LAB_FEATURES,
        cfg.LAB_FEATURES
    )

    if df_full is None:
        print("数据加载或预处理失败，程序终止。")
        return

    # 准备特征 (X) 和标签 (y)
    X_structured = df_full[cfg.STRUCTURED_FEATURES]
    y = df_full[cfg.LABEL_COLUMN]

    # 获取非结构化特征 (hu) 和置信度 (pu)
    hu_cols = list(range(cfg.DU))
    pu_col = cfg.DU

    if not hu_cols or pu_col not in df_full.columns:
        print(f"错误: 找不到非结构化特征列 (应以 'feature_' 开头) 或置信度列 ('{pu_col}')")
        print("     请检查 UNSTRUCTURED_FEAT_CSV 文件中的列名。")
        return

    X_unstructured_hu = df_full[hu_cols].values
    unstructured_pu = df_full[pu_col].values

    # t-SNE 可视化 (原始结构化数据)
    print(X_structured.shape)
    X_structured_scaled = StandardScaler().fit_transform(X_structured)
    utils.plot_tsne(X_structured_scaled, y.values, "t-SNE of Original Structured Data", "tsne_original.png")

    # 2. K-Fold 交叉验证
    skf = StratifiedKFold(n_splits=cfg.K_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    best_models = []  # 保存每个fold的最佳模型信息

    for fold, (train_index, test_index) in enumerate(skf.split(X_structured, y)):
        print(f"\n--- 开始 Fold {fold + 1}/{cfg.K_FOLDS} ---")

        # 划分数据
        X_s_train, X_s_test = X_structured.iloc[train_index], X_structured.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_u_train, X_u_test = X_unstructured_hu[train_index], X_unstructured_hu[test_index]
        pu_train, pu_test = unstructured_pu[train_index], unstructured_pu[test_index]

        # 3. 拟合结构化特征提取器
        print("拟合结构化特征提取器...")
        extractor = StructuredFeatureExtractor()
        extractor.fit(X_s_train, y_train)

        # 4. 转换训练集和测试集
        hs_train, ps_train, _ = extractor.transform(X_s_train)
        hs_test, ps_test, _ = extractor.transform(X_s_test)

        # t-SNE (处理后的训练集)
        if fold == 0:
            utils.plot_tsne(hs_train, y_train.values, f"t-SNE of Processed Structured Data (Fold 0)",
                            "tsne_processed_fold0.png")

        # 5. 创建 Dataset 和 DataLoader
        train_dataset = RADataset(hs_train, X_u_train, ps_train, pu_train, y_train.values)
        test_dataset = RADataset(hs_test, X_u_test, ps_test, pu_test, y_test.values)

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

        # 6. 初始化模型和损失函数
        model = CHAN_Model(ds=hs_train.shape[1], du=X_u_train.shape[1]).to(cfg.DEVICE)
        criterion_ce = nn.BCELoss()  # (L_CE)
        criterion_contrast = CA_OCL_Loss(device=cfg.DEVICE)  # (L_contrast)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_metrics = None

        # 7. 训练循环
        for epoch in range(cfg.EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_contrast,
                                         cfg.GAMMA_CONTRAST, cfg.DEVICE)
            val_loss, metrics, contra_acc = evaluate(model, test_loader, criterion_ce, criterion_contrast,
                                                     cfg.GAMMA_CONTRAST, cfg.DEVICE)

            acc, prec, rec, f1 = metrics

            print(
                f"Epoch {epoch + 1}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | ContraAcc: {contra_acc:.4f}")

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                best_metrics = metrics + (contra_acc,)
            else:
                patience_counter += 1
                if patience_counter >= cfg.EARLY_STOP_PATIENCE:
                    print("早停触发。")
                    break

        # 保存最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            save_model(model, extractor, fold, model_dir)
            best_models.append({
                'fold': fold,
                'model_state': best_model_state,
                'extractor': extractor,
                'metrics': best_metrics
            })

        # 记录该折的最佳结果
        if best_metrics is not None:
             fold_results.append(best_metrics)

    # 8. 汇总结果
    results_df = pd.DataFrame(fold_results,
                              columns=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Contradictory_Acc'])
    print("\n--- 交叉验证最终结果 (平均值) ---")
    print(results_df.mean())

    # 保存交叉验证结果
    results_df.to_csv(os.path.join(model_dir, "cross_validation_results.csv"), index=False)
    print(f"\n模型已保存到 {model_dir} 目录")
    print(f"交叉验证结果已保存到 {os.path.join(model_dir, 'cross_validation_results.csv')}")


if __name__ == "__main__":
    main()