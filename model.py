# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg


# --- 辅助模块：前馈网络 (FFN) ---

class FeedForward(nn.Module):
    """
    一个标准的前馈网络模块 (FFN)，用于 Transformer 模块中。
    Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4  # 常见的 FFN 扩展倍数
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


# --- 3.1.4 CHAN 模块 ---

class IntraModalSelfAttention(nn.Module):
    """
    (丰富版) (式 14-16) 模态内自注意力增强

    原始实现仅为 (Attention -> Add & Norm)。
    这里我们将其丰富为一个完整的 Transformer 编码器模块：
    (Attention -> Add & Norm -> FeedForward -> Add & Norm)
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 1. 自注意力层 (式 14-15)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        # 将 sqrt(dk) 注册为 buffer，避免重复计算
        self.register_buffer('dk_sqrt', torch.tensor(d_model).float().sqrt())

        # 2. 前馈网络 (FFN)
        self.ffn = FeedForward(d_model, dropout=dropout)

        # 3. Add & Norm
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, ds)
        # 自注意力需要序列，我们将 (B, ds) 视为 (B, 1, ds)
        x_seq = x.unsqueeze(1)  # (B, 1, ds)

        # --- 1. (Attention) ---
        Qs = self.Wq(x_seq)  # (B, 1, ds)
        Ks = self.Wk(x_seq)  # (B, 1, ds)
        Vs = self.Wv(x_seq)  # (B, 1, ds)

        # (式 15)
        attn_weights = F.softmax(torch.bmm(Qs, Ks.transpose(1, 2)) / self.dk_sqrt, dim=-1)
        attention = torch.bmm(attn_weights, Vs)  # (B, 1, ds)

        # --- 2. (Add & Norm 1) --- (式 16 基础)
        x_attn = self.layer_norm1(x_seq + self.dropout1(attention))

        # --- 3. (FeedForward) ---
        x_ffn = self.ffn(x_attn)

        # --- 4. (Add & Norm 2) ---
        x_enhanced = self.layer_norm2(x_attn + self.dropout2(x_ffn))

        return x_enhanced.squeeze(1)  # (B, ds)


class CrossModalAttention(nn.Module):
    """(式 17-18) 跨模态注意力对齐 """

    def __init__(self, d_query, d_kv):
        super().__init__()
        self.Wq = nn.Linear(d_query, d_query, bias=False)
        self.Wk = nn.Linear(d_kv, d_query, bias=False)
        self.Wv = nn.Linear(d_kv, d_query, bias=False)

        # 将 sqrt(dk) 注册为 buffer
        self.register_buffer('dk_sqrt', torch.tensor(d_query).float().sqrt())

    def forward(self, query, kv, confidence_bias):
        # query: (B, d_query)
        # kv: (B, d_kv)
        # confidence_bias: (B, 1)

        # (B, 1, d_query)
        Q = self.Wq(query.unsqueeze(1))
        # (B, 1, d_query)
        K = self.Wk(kv.unsqueeze(1))
        # (B, 1, d_query)
        V = self.Wv(kv.unsqueeze(1))

        # (式 17)
        # confidence_bias (ps or pu) 必须 > 0 才能取 log
        # 我们使用 log(ps + 1e-9) 来保证数值稳定性
        log_bias = torch.log(confidence_bias.unsqueeze(1) + 1e-9)

        alpha = F.softmax((torch.bmm(Q, K.transpose(1, 2)) / self.dk_sqrt) + log_bias, dim=-1)

        # (式 18)
        h_aligned = torch.bmm(alpha, V)  # (B, 1, d_query)

        return h_aligned.squeeze(1)


class CHAN_Model(nn.Module):
    """
    完整的 CHAN 模型 (3.1.4) 和 CA-OCL 投影头 (3.1.3)
    """

    def __init__(self, ds=cfg.DS, du=cfg.DU, projection_dim=cfg.PROJECTION_DIM, dropout=0.1):
        super().__init__()

        self.ds = ds
        self.du = du

        # --- 3.1.3 CA-OCL 投影头 ---
        # 将 hs 和 hu 映射到共享语义空间
        self.projection_s = nn.Linear(ds, projection_dim)
        self.projection_u = nn.Linear(du, projection_dim)

        # --- 3.1.4 CHAN 模块 ---

        # 1. 模态内自注意力
        self.intra_s_attention = IntraModalSelfAttention(d_model=ds, dropout=dropout)

        # 2. 跨模态对齐
        # 结构化 -> 非结构化 (hs2u)
        self.cross_s2u_attention = CrossModalAttention(d_query=ds, d_kv=du)
        # 非结构化 -> 结构化 (hu2s)
        self.cross_u2s_attention = CrossModalAttention(d_query=du, d_kv=ds)

        # 3. 门控自适应融合
        self.gating_layer = nn.Linear(2, 1)  # 输入是 [p1; p2]

        self.Wprojs = nn.Linear(ds, du)

        # 4. 分类器
        # (Linear -> ReLU -> Dropout -> Linear)
        self.classifier = nn.Sequential(
            nn.Linear(du, du // 2),
            nn.LayerNorm(du // 2),  # 增加 LayerNorm 提高稳定性
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(du // 2, 1)
        )

    def forward(self, hs, hu, ps, pu):
        # hs: (B, ds) 结构化特征
        # hu: (B, du) 非结构化特征
        # ps: (B, 1) 结构化置信度
        # pu: (B, 1) 非结构化置信度

        # --- CA-OCL 投影  ---
        Zs = self.projection_s(hs)
        Zu = self.projection_u(hu)

        # --- CHAN 流程 ---

        # 1. 模态内增强
        hs_intra = self.intra_s_attention(hs)  # (B, ds) (丰富版)
        hu_intra = hu  # (B, du) (直接重用)

        # 2. 跨模态对齐
        hs2u = self.cross_s2u_attention(hs_intra, hu_intra, ps)  # (B, ds)
        hu2s = self.cross_u2s_attention(hu_intra, hs_intra, pu)  # (B, du)

        # 3. 门控融合
        p_concat = torch.cat([ps, pu], dim=1)  # (B, 2)
        g = torch.sigmoid(self.gating_layer(p_concat))  # (B, 1) (式 19)

        hs2u_projected = self.Wprojs(hs2u)  # (B, du)

        hf = g * hs2u_projected + (1.0 - g) * hu2s

        # 4. 分类
        logits = self.classifier(hf)  # (B, 1)
        p = torch.sigmoid(logits)  # (B, 1)

        # 返回 p (用于BCE loss), Zs, Zu (用于CA-OCL loss)
        # 同时也返回投影层模块，以便 L_orth 计算
        return p, Zs, Zu, self.projection_s, self.projection_u


# --- 3.1.3 CA-OCL 损失函数 ---

class CA_OCL_Loss(nn.Module):
    def __init__(self, tau=cfg.TAU, alpha=cfg.CONTR_ALPHA, lambda_c=cfg.CONTR_LAMBDA, beta=cfg.BETA_ORTH,
                 device=cfg.DEVICE):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.lambda_c = lambda_c
        self.beta = beta
        self.device = device

    def forward(self, Zs, Zu, projection_s, projection_u, ps, pu, labels):
        """
        计算 CA-OCL 损失 (式 13)
        Args:
            Zs (torch.Tensor): 结构化投影特征 (B, projection_dim)
            Zu (torch.Tensor): 非结构化投影特征 (B, projection_dim)
            projection_s (nn.Linear): 结构化投影层模块 (用于 L_orth)
            projection_u (nn.Linear): 非结构化投影层模块 (用于 L_orth)
            ps (torch.Tensor): 结构化置信度 (B, 1)
            pu (torch.Tensor): 非结构化置信度 (B, 1)
            labels (torch.Tensor): 真实标签 (B, 1)
        """

        B = Zs.shape[0]
        if B < 2:
            # 样本太少无法计算对比损失
            return torch.tensor(0.0).to(self.device)

            # 归一化
        Zs_norm = F.normalize(Zs, p=2, dim=1)
        Zu_norm = F.normalize(Zu, p=2, dim=1)

        # --- 1. 基础损失 (L_base)---
        sim_matrix = torch.mm(Zs_norm, Zu_norm.t()) / self.tau

        positive_pairs = torch.diag(sim_matrix)

        L_base = -positive_pairs + torch.logsumexp(sim_matrix, dim=1)
        L_base = L_base.mean()

        # --- 2. 矛盾样本加权损失 (L_contr)  ---

        # "C" 是矛盾负样本集 (Contradictory Negative Samples)
        # 寻找 (i, j) 对，满足:
        #   1. i != j (负样本)
        #   2. label[i] == label[j] (共享相同诊断)
        #   3. sample[i] 和 sample[j] 的矛盾状态不同
        #      (一个和谐，一个矛盾)

        # 识别哪些样本本身是矛盾的 (ps 和 pu 预测不一致)
        ps_pred = (ps > 0.5).float()
        pu_pred = (pu > 0.5).float()
        sample_is_contradictory = (ps_pred != pu_pred).float()  # (B, 1)

        # (B, B) 矩阵，ij=1 表示 (i, j) 共享相同标签
        labels_matrix = (labels == labels.t()).float()

        # (B, B) 矩阵，ij=1 表示 (i, j) 矛盾状态不同
        contradictory_status_matrix = (sample_is_contradictory != sample_is_contradictory.t()).float()

        # 负样本掩码 (非对角线)
        neg_mask = 1.0 - torch.eye(B, device=self.device)

        # C_mask: (B, B) 矩阵，Cij=1 表示 (i,j) 是矛盾负样本
        C_mask = neg_mask * labels_matrix * contradictory_status_matrix

        if C_mask.sum() > 0:
            # (式 11)
            # Zs_norm (B, D), Zu_norm.t() (D, B)
            cosine_sim_neg = torch.mm(Zs_norm, Zu_norm.t())
            # (1 - cos)
            contr_loss_matrix = 1.0 - cosine_sim_neg

            # (alpha * lambda) * (1 - cos) * C_mask
            # 只在 C_mask 为 1 的地方计算损失
            L_contr = self.alpha * self.lambda_c * (contr_loss_matrix * C_mask)
            # 归一化
            L_contr = L_contr.sum() / (C_mask.sum() + 1e-9)
        else:
            L_contr = torch.tensor(0.0).to(self.device)

        # --- 3. 正交约束损失 (L_orth) (Eq 12) ---
        # 约束投影矩阵 Ws 和 Wu
        Ws_proj = projection_s.weight
        Wu_proj = projection_u.weight

        WtW = torch.mm(Ws_proj.t(), Wu_proj)
        L_orth = torch.norm(WtW, p='fro') ** 2

        total_loss = L_base + L_contr + self.beta * L_orth

        return total_loss