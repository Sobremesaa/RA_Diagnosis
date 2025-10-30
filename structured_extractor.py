import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import gaussian_kde
import warnings


class StructuredFeatureExtractor:
    """
    实现 3.1.1 节中的结构化特征处理
    - 异常值检测 (IQR)
    - 动态权重分配 (基于核密度估计的MI)
    - 轻量级特征交叉
    - 置信度分数 (LR)
    """

    def __init__(self, k_outlier=2, k_cross=2, lambda_attenuation=1.0):
        """
        初始化

        Parameters:
        -----------
        k_outlier : int
            选择的异常特征数量 (论文中设为2)
        k_cross : int
            选择的交叉特征数量 (论文中设为2)
        lambda_attenuation : float
            衰减系数 (论文中未明确指定，设为1.0)
        """
        self.k_outlier = k_outlier
        self.k_cross = k_cross
        self.lambda_attenuation = lambda_attenuation

        self.scaler = StandardScaler()
        self.lr_model_ps = LogisticRegression(solver='liblinear')  # 置信度分数模型
        self.lr_model_wij = LinearRegression()  # 用于学习 wij

        # IQR 相关属性
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self.outlier_thresholds = None
        self.top_k_outlier_indices = None

        # MI 相关属性
        self.mi_weights = None
        self.initial_mi_weights = None  # 初始MI权重

        # 特征交叉相关属性
        self.top_k_cross_indices = None
        self.poly_feature_names = None
        self.wij = None  # 交叉权重

        self.feature_names_in_ = None
        self.poly = None  # 多项式特征转换器

    def fit(self, X_df, y):
        """
        使用训练数据拟合提取器

        Parameters:
        -----------
        X_df : DataFrame
            结构化特征的 DataFrame (e.g., age, gender, CRP, RF, ACPA, ESR)
        y : Series
            标签 Series
        """
        self.feature_names_in_ = X_df.columns.tolist()
        X = self.scaler.fit_transform(X_df)

        # 1. 异常值检测 (IQR) - 公式(1)(2)
        self._fit_iqr(X)

        # 2. 动态权重分配 (基于核密度估计的MI) - 公式(3)-(6)
        self._fit_mi_weights(X, y)

        # 3. 轻量级特征交叉 - 公式(8)
        self._fit_feature_crossing(X, y)

        # 4. 构建特征矩阵 hs - 公式(9)
        hs = self._build_hs_matrix(X)

        # 5. 训练置信度分数模型
        self.lr_model_ps.fit(hs, y)
        print("结构化特征提取器拟合完毕。")

    def _fit_iqr(self, X):
        """拟合IQR异常检测 - 公式(1)(2)"""
        # 计算四分位数和IQR
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1  # 公式(1)

        # 计算异常阈值 - 公式(2)
        lower_bound = self.q1 - 1.5 * self.iqr
        upper_bound = self.q3 + 1.5 * self.iqr
        self.outlier_thresholds = (lower_bound, upper_bound)

        # 计算每个特征的异常比例并选择top K
        outlier_mask = (X < lower_bound) | (X > upper_bound)
        outlier_proportions = np.mean(outlier_mask, axis=0)
        self.top_k_outlier_indices = np.argsort(outlier_proportions)[-self.k_outlier:]

        print(f"Top {self.k_outlier} 异常特征索引: {self.top_k_outlier_indices}")
        print(f"对应特征: {[self.feature_names_in_[i] for i in self.top_k_outlier_indices]}")

    def _compute_mi_with_kde(self, X, y):
        """
        使用核密度估计计算互信息 - 公式(3)-(6)
        """
        n_features = X.shape[1]
        mi_scores = np.zeros(n_features)

        # 使用sklearn的MI作为备选，因为KDE方法可能不稳定
        mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

        return mi_scores

    def _fit_mi_weights(self, X, y):
        """拟合互信息权重"""
        # 使用KDE-based MI计算
        try:
            mi_scores = self._compute_mi_with_kde(X, y)
        except:
            # 如果KDE方法失败，使用sklearn的MI作为备选
            warnings.warn("KDE-based MI计算失败，使用sklearn的MI作为备选")
            mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)

        # 归一化MI分数作为初始权重
        self.initial_mi_weights = mi_scores / (np.sum(mi_scores) + 1e-9)
        self.mi_weights = self.initial_mi_weights.copy()

        print("初始MI权重:", dict(zip(self.feature_names_in_, self.initial_mi_weights)))

    def _fit_feature_crossing(self, X, y):
        """拟合特征交叉 - 公式(8)"""
        # 生成多项式特征（仅交互项）
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.poly_feature_names = self.poly.get_feature_names_out(self.feature_names_in_)

        # 仅保留交互项
        interaction_indices = [i for i, name in enumerate(self.poly_feature_names) if ' ' in name]
        X_interactions = X_poly[:, interaction_indices]

        # 使用最小二乘法学习 wij - 公式(8)
        self.lr_model_wij.fit(X_interactions, y)
        self.wij = self.lr_model_wij.coef_

        # 选择 K 个最相关的交叉组合 (基于 wij 的绝对值)
        top_k_indices_local = np.argsort(np.abs(self.wij))[-self.k_cross:]
        self.top_k_cross_indices = [interaction_indices[i] for i in top_k_indices_local]

        print(f"Top {self.k_cross} 交叉特征:")
        for idx in self.top_k_cross_indices:
            print(f"  {self.poly_feature_names[idx]}: wij = {self.wij[interaction_indices.index(idx)]:.4f}")

    def _build_hs_matrix(self, X):
        """构建特征矩阵 hs - 公式(9)"""
        n_samples, n_features = X.shape

        # 1. 获取异常矩阵 O
        lower_bound, upper_bound = self.outlier_thresholds
        outlier_mask = (X < lower_bound) | (X > upper_bound)
        O_matrix = outlier_mask[:, self.top_k_outlier_indices].astype(float)

        # 2. 计算动态权重 β - 公式(7)
        # 对于每个样本，如果在任何选定的异常特征上有异常，则衰减权重
        O_flag_per_sample = np.any(outlier_mask[:, self.top_k_outlier_indices], axis=1).astype(float)

        # 扩展为与特征数相同的维度
        O_flag_expanded = np.tile(O_flag_per_sample.reshape(-1, 1), (1, n_features))

        # 应用衰减 - 公式(7)
        beta_matrix = np.tile(self.initial_mi_weights, (n_samples, 1)) * \
                      (1.0 - O_flag_expanded * self.lambda_attenuation)

        # 3. 获取加权特征 X ⊙ β
        X_weighted = X * beta_matrix

        # 4. 获取交叉特征 X_cross - 公式(8)
        X_poly = self.poly.transform(X)  # 使用已拟合的多项式转换器
        X_cross = X_poly[:, self.top_k_cross_indices]

        # 5. 组合 hs - 公式(9)
        hs = np.column_stack([X_weighted, X_cross, O_matrix])

        print(f"构建hs矩阵: X_weighted.shape={X_weighted.shape}, "
              f"X_cross.shape={X_cross.shape}, O_matrix.shape={O_matrix.shape}, "
              f"hs.shape={hs.shape}")

        return hs

    def transform(self, X_df):
        """
        使用已拟合的参数转换数据

        Parameters:
        -----------
        X_df : DataFrame
            输入特征数据

        Returns:
        --------
        hs : ndarray
            特征矩阵
        ps : ndarray
            置信度分数
        O_matrix : ndarray
            异常矩阵
        """
        if self.feature_names_in_ is None:
            raise NotFittedError("Extractor尚未拟合(fit)。")

        X_df = X_df[self.feature_names_in_]
        X = self.scaler.transform(X_df)

        # 构建特征矩阵
        hs = self._build_hs_matrix(X)

        # 获取异常矩阵
        lower_bound, upper_bound = self.outlier_thresholds
        outlier_mask = (X < lower_bound) | (X > upper_bound)
        O_matrix = outlier_mask[:, self.top_k_outlier_indices].astype(float)

        # 预测置信度分数
        ps = self.lr_model_ps.predict_proba(hs)[:, 1]

        return hs, ps, O_matrix