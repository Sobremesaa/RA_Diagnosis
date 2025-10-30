import pandas as pd
import numpy as np
import config as cfg


def load_data(structured_csv_path, unstructured_csv_path):
    """
    加载原始结构化数据(456行)和处理后的非结构化特征(912行)
    """
    try:
        df_structured_raw = pd.read_csv(structured_csv_path)
        print(f"加载原始结构化数据: {structured_csv_path} ({len(df_structured_raw)} 行)")
    except FileNotFoundError:
        print(f"错误: 原始结构化数据文件未找到: {structured_csv_path}")
        return None, None

    try:
        df_unstructured_feats = pd.read_csv(unstructured_csv_path, header=None)
        print(f"加载非结构化特征: {unstructured_csv_path} ({len(df_unstructured_feats)} 行)")
    except FileNotFoundError:
        print(f"错误: 非结构化特征文件未找到: {unstructured_csv_path}")
        return None, None

    # 数据增强使数据集加倍 (456 -> 912)
    expected_rows = len(df_structured_raw) * 2
    if len(df_unstructured_feats) != expected_rows:
        print(f"!! 警告: 你的非结构化特征有 {len(df_unstructured_feats)} 行。")
        print(f"   基于结构化数据 {len(df_structured_raw)} 行, 增强后应为 {expected_rows} 行。")
        print(f"   请确保你的 {unstructured_csv_path} 包含 {expected_rows} 行，")
        print("   并且顺序与增强后的结构化数据 (原始1-456 + 增强的457-912) 一致。")
        if len(df_unstructured_feats) < expected_rows:
            print("   错误：行数不匹配，无法继续。")
            return None, None

    return df_structured_raw, df_unstructured_feats


def handle_transformations(df):
    """
    处理数据转换
    """
    df_proc = df.copy()

    # 1. 处理性别
    if 'gender' in df_proc.columns:
        df_proc['gender'] = df_proc['gender'].apply(lambda x: 1 if str(x).lower() in ['female', '女'] else 0)

    # 2. 处理CRP - 阴性范围: 0<=CRP<=6
    if 'CRP' in df_proc.columns:
        negative_mask = df_proc['CRP'].astype(str).str.lower().isin(['阴性', 'negative'])
        num_negatives = negative_mask.sum()
        if num_negatives > 0:
            random_values = np.random.uniform(0, 6, num_negatives)
            df_proc.loc[negative_mask, 'CRP'] = random_values
        df_proc['CRP'] = pd.to_numeric(df_proc['CRP'], errors='coerce')

    # 3. 处理RF - 阴性范围: 0-20
    if 'RF' in df_proc.columns:
        negative_mask = df_proc['RF'].astype(str).str.lower().isin(['阴性', 'negative'])
        num_negatives = negative_mask.sum()
        if num_negatives > 0:
            random_values = np.random.uniform(0, 20, num_negatives)
            df_proc.loc[negative_mask, 'RF'] = random_values
        df_proc['RF'] = pd.to_numeric(df_proc['RF'], errors='coerce')

    # 4. 处理ACPA (Anti-CCP) - 阴性范围: <5.0
    if 'ACPA' in df_proc.columns:
        # 先处理阴性值
        negative_mask = df_proc['ACPA'].astype(str).str.lower().isin(['阴性', 'negative'])
        num_negatives = negative_mask.sum()
        if num_negatives > 0:
            random_values = np.random.uniform(0, 5.0, num_negatives)
            df_proc.loc[negative_mask, 'ACPA'] = random_values

        df_proc['ACPA'] = df_proc['ACPA'].astype(str)
        df_proc['ACPA'] = df_proc['ACPA'].str.replace('<', '', regex=False).str.replace('>', '', regex=False)
        df_proc['ACPA'] = pd.to_numeric(df_proc['ACPA'], errors='coerce')

        df_proc.loc[df_proc['ACPA'] < 0.5, 'ACPA'] = 0.5
        df_proc.loc[df_proc['ACPA'] > 200.0, 'ACPA'] = 200.0

    # 5. 处理ESR - 根据性别确定阴性范围
    if 'ESR' in df_proc.columns:
        negative_mask = df_proc['ESR'].astype(str).str.lower().isin(['阴性', 'negative'])
        num_negatives = negative_mask.sum()

        if num_negatives > 0:
            esr_values = []
            for idx in df_proc[negative_mask].index:
                gender = df_proc.loc[idx, 'gender'] if 'gender' in df_proc.columns else 0
                if gender == 1:  # 女性
                    esr_values.append(np.random.uniform(0, 20))
                else:  # 男性
                    esr_values.append(np.random.uniform(0, 15))

            df_proc.loc[negative_mask, 'ESR'] = esr_values

        df_proc['ESR'] = pd.to_numeric(df_proc['ESR'], errors='coerce')

    return df_proc


def impute_missing_values(df, features, label_col):
    """
    使用经验分布进行分层随机抽样来填充缺失值
    """
    df_imputed = df.copy()

    for feature in features:
        if df_imputed[feature].isnull().sum() == 0:
            continue

        for label in df_imputed[label_col].unique():
            label_mask = (df_imputed[label_col] == label)
            missing_mask = df_imputed[feature].isnull() & label_mask

            if missing_mask.sum() == 0:
                continue

            observed_values = df_imputed.loc[label_mask & ~missing_mask, feature]

            if observed_values.empty:
                fill_values = df_imputed[feature].mean()
            else:
                fill_values = np.random.choice(observed_values.values, size=missing_mask.sum(), replace=True)

            df_imputed.loc[missing_mask, feature] = fill_values

    return df_imputed


def augment_data(df, features, label_col):
    """
    通过随机替换进行数据增强(返回 2*N 条数据)
    """
    df_augmented_list = [df]  # 保留原始数据 (1-456)

    for label in df[label_col].unique():
        df_label_group = df[df[label_col] == label].copy()
        if df_label_group.empty:
            continue

        df_new_samples = df_label_group.copy()

        for feature in features:
            observed_values = df_label_group[feature].values
            if len(observed_values) > 0:
                random_replacements = np.random.choice(observed_values, size=len(df_new_samples), replace=True)
                df_new_samples[feature] = random_replacements

        df_augmented_list.append(df_new_samples)  # 添加增强数据 (457-912)

    df_augmented = pd.concat(df_augmented_list, ignore_index=True)
    print(f"数据增强完成。 样本数从 {len(df)} 增加到 {len(df_augmented)}")
    return df_augmented


def preprocess_main(structured_csv_path, unstructured_csv_path, label_col, features_to_process, features_to_augment):
    """
    完整的数据预处理流程：
    1. 加载 456 行结构化 和 912 行非结构化数据
    2. 处理 456 行结构化数据 (转换、填充)
    3. 增强 456 -> 912 行结构化数据
    4. 按行号合并 912 行结构化数据 和 912 行非结构化数据
    """

    # 1. 加载数据
    df_structured_raw, df_unstructured_feats = load_data(structured_csv_path, unstructured_csv_path)
    if df_structured_raw is None:
        return None

    # 2. 数据转换 (在 456 行上)
    df_transformed = handle_transformations(df_structured_raw)

    # 3. 缺失值填充 (在 456 行上)
    df_imputed = impute_missing_values(df_transformed, features_to_process, label_col)

    # 4. 数据增强 (456 -> 912 行)
    df_structured_augmented = augment_data(df_imputed, features_to_augment, label_col)

    if len(df_structured_augmented) != len(df_unstructured_feats):
        print(f"!! 严重错误: 增强后的结构化数据有 {len(df_structured_augmented)} 行,")
        print(f"     但你提供的非结构化特征数据有 {len(df_unstructured_feats)} 行。")
        print("     两者必须具有相同的行数才能按行号对应。请检查你的数据。")
        return None

    df_structured_augmented = df_structured_augmented.reset_index(drop=True)
    df_unstructured_feats = df_unstructured_feats.reset_index(drop=True)
    print(df_structured_augmented.head())
    print(df_unstructured_feats)

    df_full = pd.concat([df_structured_augmented, df_unstructured_feats], axis=1)
    df_full[label_col] = np.where(df_full[label_col] == '类风湿关节炎', 1, 0)
    print(df_full[label_col])
    print(f"数据已按行号合并。总样本数: {len(df_full)}")

    df_full[label_col] = pd.to_numeric(df_full[label_col], errors='coerce')
    df_full = df_full.dropna(subset=[label_col])
    print(df_full)

    return df_full