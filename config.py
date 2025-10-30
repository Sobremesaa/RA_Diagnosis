import torch

# --- 路径配置 ---
STRUCTURED_DATA_CSV = "D:/master/code/RA_Diagnosis/data/original_data.csv"
UNSTRUCTURED_FEAT_CSV = "D:/master/code/RA_Diagnosis/data/bert_features.csv"
STRUCTURED_TEST_CSV = "D:/master/code/RA_Diagnosis/data/test_data.csv"
UNSTRUCTURED_TEST_CSV = "D:/master/code/RA_Diagnosis/data/bert_test.csv"

MODEL_DIR="models"
FOLD_TO_USE = 1


# --- 数据处理配置 (3.1.1节) ---
# 用于IQR异常值检测 (K=2)
OUTLIER_K_FEATURES = 2
# 特征交叉中选择最相关的组合数量
CROSS_FEATURE_K_COMBINATIONS = 2
# 结构化数据中的关键实验室指标
LAB_FEATURES = ['CRP', 'RF', 'ACPA', 'ESR']
STRUCTURED_FEATURES = ['age', 'gender'] + LAB_FEATURES
LABEL_COLUMN = "label"

# --- 模型维度 (3.1.4节) ---
# 结构化特征 hs 的维度
DS = 10
# 非结构化特征 hu 的维度
DU = 32
# CA-OCL 共享语义空间的维度
PROJECTION_DIM = 128

# --- 训练配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-5
K_FOLDS = 5
EARLY_STOP_PATIENCE = 3

# --- CA-OCL 损失超参数 ---
# 温度系数 τ
TAU = 0.1
# 矛盾样本的惩罚权重 λ (alpha)
CONTR_ALPHA = 2.0
# 平衡系数 λ (lambda)
CONTR_LAMBDA = 1.0
# 正交约束强度 β
BETA_ORTH = 0.05

# --- CHAN 总损失超参数 ---
# 均衡系数 γ
GAMMA_CONTRAST = 0.3