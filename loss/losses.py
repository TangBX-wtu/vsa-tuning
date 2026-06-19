import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """标准交叉熵，支持类别不平衡的 label smoothing"""

    def __init__(self, num_classes: int, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, labels)


class ProbeLoss(nn.Module):
    """
    探针引导损失：让模型的 attention 权重与弱标注掩码 M 对齐。

    使用 KL 散度：
        L_probe = KL( softmax(M/τ) || attention_weights )

    直觉：弱标注掩码 M 是一个"软目标"，告诉模型"应该关注这里"。
    τ（温度）控制目标分布的尖锐程度：
        τ 小 → 强制模型只关注关键片段（可能过拟合）
        τ 大 → 目标分布接近均匀，监督信号弱
    τ=0.5 默认，2.0目标分布太软。
    当某个样本的 key_mask 全为 0（无弱标注）时，目标 softmax
    会产生均匀分布而非 NaN，并对这类样本单独处理以避免污染整体损失。
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(
            self,
            attn_weights: torch.Tensor,  # (B, L)  模型的 attention 权重
            key_mask: torch.Tensor,  # (B, L)  弱标注掩码（0/1 或连续值）
            padding_mask: torch.Tensor,  # (B, L)  1=真实 token
    ) -> torch.Tensor:
        neg_inf = torch.finfo(attn_weights.dtype).min

        # 目标分布：弱标注掩码经 softmax 软化
        target = key_mask.float()
        # 先把 padding 位置填为 neg_inf
        target = target.masked_fill(padding_mask == 0, neg_inf)

        # 检测哪些样本的真实 token 区域内 key_mask 全为 0
        # 对这类样本，softmax 输入全 0 → 均匀分布（合理的 fallback），不会 NaN
        # 但若真实 token 数为 0（极端情况），则整行都是 neg_inf → softmax NaN
        # 用下面的 clamp 保证至少有一个位置不是 neg_inf
        has_real_token = (padding_mask == 1).any(dim=-1)  # (B,)
        if not has_real_token.all():
            # 极端情况：某个样本完全没有真实 token，跳过整个 batch 的 probe loss
            return torch.tensor(0.0, device=attn_weights.device, requires_grad=False)

        target_dist = F.softmax(target / self.tau, dim=-1)  # (B, L)

        # 检查 target_dist 是否含 NaN
        if torch.isnan(target_dist).any():
            return torch.tensor(0.0, device=attn_weights.device, requires_grad=False)

        # 模型 attention 分布（log 空间）
        # clamp attn_weights 防止 log(0)
        pred_log = torch.log(attn_weights.clamp(min=1e-8))  # (B, L)

        # KL(target || pred)
        # loss = F.kl_div(pred_log, target_dist, reduction="batchmean",)
        loss = F.kl_div(
            torch.log(target_dist.clamp(min=1e-8)),
            attn_weights.clamp(min=1e-8),
            reduction="batchmean",
        )

        # 最终 NaN 检查
        if torch.isnan(loss):
            return torch.tensor(0.0, device=attn_weights.device, requires_grad=False)

        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss（SupCon）。

    参考：Khosla et al., 2020, "Supervised Contrastive Learning"

    正样本对定义：
      - 同类漏洞类型的不同代码实例互为正样本
      - 无漏洞样本（类别 0）之间互为正样本

    关键设计：
      漏洞类别 0（无漏洞）和漏洞类别 1-N 在对比空间中应该自然分离，
      这正是我们希望"漏洞语义空间"具备的性质。
      1. 当 batch 中所有样本都是孤立类别时，返回真正的零梯度张量而非 detach 的常量，避免 NaN 传播。
      2. log_softmax 前对 sim_matrix 做数值稳定性检查。
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(
            self,
            embeddings: torch.Tensor,  # (B, D)  L2 归一化后的投影表示
            labels: torch.Tensor,  # (B,)    类别标签
    ) -> torch.Tensor:
        B = embeddings.size(0)
        device = embeddings.device

        # ── 临时诊断，确认后删除 ──
        # print(f"[Con诊断] B={B}, labels={labels.tolist()}")
        # eye_mask = torch.eye(B, dtype=torch.bool, device=device)
        # pos_mask_check = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye_mask
        # pos_count_check = pos_mask_check.float().sum(dim=1)
        # print(f"[Con诊断] pos_count per sample: {pos_count_check.tolist()}")
        # print(f"[Con诊断] has_pos: {(pos_count_check > 0).tolist()}")
        # ─────────────────────────

        # 强制转为 int64，防止 float/one-hot 导致 == 比较失效
        labels = labels.long()
        # L2 归一化（如果传入前未归一化，在这里做）
        z = F.normalize(embeddings, dim=-1)  # (B, D)

        # 相似度矩阵，先不除温度
        sim_matrix = torch.matmul(z, z.T)  # (B, B)，值域 [-1, 1]

        # 排除对角线（自身与自身的相似度）
        eye_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(eye_mask, float("-inf"))

        # 除温度
        sim_matrix = sim_matrix / self.tau  # 现在最大值约 1/tau

        # 数值稳定：每行减去该行最大值，再做 log_softmax
        # 这等价于标准的 log_softmax stable trick，防止 exp 溢出
        # 注意：masked_fill(-inf) 的位置不参与 max 计算
        max_val = sim_matrix.detach().masked_fill(eye_mask, float("-inf")).max(dim=1, keepdim=True).values
        max_val = max_val.clamp(min=0)  # 防止全 -inf 行
        sim_stable = sim_matrix - max_val  # 平移后最大值为 0，exp 最大为 1

        log_prob = F.log_softmax(sim_stable, dim=1)  # (B, B)
        '''
        # 临时诊断
        print(f"[NaN诊断] sim_matrix stats: min={sim_matrix[~eye_mask].min().item():.3f} "
              f"max={sim_matrix[~eye_mask].max().item():.3f}")
        print(f"[NaN诊断] max_val stats: min={max_val.min().item():.3f} max={max_val.max().item():.3f}")
        print(f"[NaN诊断] sim_stable stats: min={sim_stable[~eye_mask].min().item():.3f} "
              f"max={sim_stable[~eye_mask].max().item():.3f}")
        print(f"[NaN诊断] sim_stable has inf: {torch.isinf(sim_stable).any().item()}")
        print(f"[NaN诊断] sim_stable has nan: {torch.isnan(sim_stable).any().item()}")
        print(f"[NaN诊断] log_prob has nan: {torch.isnan(log_prob).any().item()}")
        nan_rows = torch.isnan(log_prob).any(dim=1)
        print(f"[NaN诊断] nan行数: {nan_rows.sum().item()}, nan行的sim_stable: "
              f"{sim_stable[nan_rows][0].tolist() if nan_rows.any() else '无'}")
        '''

        # 构造正样本掩码：同类别且不是自身
        label_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B,1) == (1,B) → (B,B)
        pos_mask = label_eq & ~eye_mask  # (B, B)

        pos_count = pos_mask.float().sum(dim=-1)  # (B,)
        has_pos = pos_count > 0

        if not has_pos.any():
            # 返回与 embeddings 计算图相连的零值，保证梯度流正常
            # 用 (z * 0).sum() 而不是 tensor(0.0)，这样 autograd 图不会断裂
            print("warning：batch中类别均为0，无法进行对比学习")
            return (z * 0.0).sum()

        '''
        # log_softmax 在全 -inf 行上可能产生 NaN，过滤掉这些行
        valid = has_pos & ~torch.isnan(log_prob).any(dim=-1)

        if not valid.any():
            print("warning：log_softmax 在全 -inf 行上产生NaN，无法进行对比学习")
            return (z * 0.0).sum()
        '''

        log_prob_safe = log_prob.masked_fill(~pos_mask, 0.0)
        loss_per = -(log_prob_safe * pos_mask.float()).sum(dim=1) / pos_count.clamp(min=1)
        '''
        # 临时诊断
        print(f"[NaN诊断2] loss_per: min={loss_per[valid].min().item():.4f} "
              f"max={loss_per[valid].max().item():.4f} "
              f"has_inf={torch.isinf(loss_per[valid]).any().item()} "
              f"has_nan={torch.isnan(loss_per[valid]).any().item()}")
        result = loss_per[valid].mean()
        print(f"[NaN诊断2] final con loss={result.item()}")
        '''
        return loss_per[has_pos].mean()


class VSALoss(nn.Module):
    """联合损失函数，组合三个子损失"""

    def __init__(self, config):
        super().__init__()
        self.cls_loss = ClassificationLoss(config.num_classes)
        self.probe_loss = ProbeLoss(config.probe_temp)
        self.con_loss = SupervisedContrastiveLoss(config.temperature)
        self.focus_loss = TriggerFocusLoss(margin=0.5)

        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.epsilon = config.epsilon

    def forward(
            self,
            logits: torch.Tensor,  # (B, num_classes)
            labels: torch.Tensor,  # (B,)
            attn_weights: torch.Tensor,  # (B, L)
            key_mask: torch.Tensor,  # (B, L)
            padding_mask: torch.Tensor,  # (B, L)
            proj_emb: torch.Tensor,  # (B, D)  对比投影表示
            trigger_scores,  # ★ 新增
            trigger_node_batch,  # ★ 新增
            trigger_subtypes,  # ★ 新增 (graph_batch.trigger_subtype)
            trigger_mask_nodes,  # ★ 新增 (graph_batch.trigger_mask)
    ) -> dict:
        L_cls = self.cls_loss(logits, labels)
        L_probe = self.probe_loss(attn_weights, key_mask, padding_mask)
        L_con = self.con_loss(proj_emb, labels)
        L_focus = self.focus_loss(
            trigger_scores, trigger_node_batch,
            trigger_subtypes, trigger_mask_nodes, labels,
        )
        '''
        print(f"[NaN诊断2] L_con={L_con.item()}, gamma={self.gamma}, "
              f"gamma*L_con={(self.gamma * L_con).item()}")
        '''
        # 逐项检查，防止单个子损失的 NaN 污染 total
        for name, L in [("cls", L_cls), ("probe", L_probe),
                        ("con", L_con), ("focus", L_focus)]:
            if torch.isnan(L):
                raise RuntimeError(f"[VSALoss] {name} 损失出现 NaN")

        L_total = (
                self.alpha * L_cls
                + self.beta * L_probe
                + self.gamma * L_con
                + self.epsilon * L_focus
        )

        return {
            "total": L_total,
            "cls": L_cls.detach(),
            "probe": L_probe.detach(),
            "contrastive": L_con.detach(),
            "focus": L_focus.detach(),  # ★ 新增
        }


class TriggerFocusLoss(nn.Module):
    """
    辅助损失:强制让"关键 subtype"的 trigger 节点得分高于其他 subtype。

    给定样本 label,查 LABEL_TO_KEY_SUBTYPE 得到该 vuln_type 应该关注的 subtype。
    在该样本的图中:
      - target trigger 节点:subtype == key_subtype 的节点
      - other trigger 节点: 其他 subtype 的 trigger 节点
    margin loss:max(0, margin - (target_mean_score - other_mean_score))

    若某样本无对应 key_subtype(NullDeref),或图中只有一种 subtype 的 trigger,
    则该样本跳过(贡献 0 损失)。
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

        # subtype id 映射(必须和 preprocess.py 里 SUBTYPE_TO_ID 对齐)
        self.SUBTYPE_FREE = 1
        self.SUBTYPE_ALLOC = 2
        self.SUBTYPE_COPY = 3
        self.SUBTYPE_INPUT = 4
        self.SUBTYPE_EXEC = 5

        # label_id → key_subtype_id(必须和 preprocess.py LABEL_MAP 对齐)
        # 0=clean, 1=UAF, 2=HBO, 3=SBO, 4=NullDeref, 5=IntOverflow
        self.LABEL_TO_KEY_SUBTYPE = {
            1: self.SUBTYPE_FREE,    # UAF
            2: self.SUBTYPE_COPY,    # HBO
            3: self.SUBTYPE_COPY,    # SBO
            5: self.SUBTYPE_ALLOC,   # IntOverflow
        }

    def forward(
        self,
        trigger_scores: torch.Tensor,         # (total_trigger_nodes,)
        trigger_node_batch: torch.Tensor,     # (total_trigger_nodes,)
        trigger_subtypes: torch.Tensor,       # (total_nodes,) 来自 graph_batch
        trigger_mask: torch.Tensor,           # (total_nodes,) 来自 graph_batch
        labels: torch.Tensor,                 # (B,)
    ) -> torch.Tensor:
        """
        实现细节:
          trigger_scores 只对应 trigger 节点(在 _build_role_query_attn 里收集),
          需要从 graph_batch.trigger_subtype 里筛选出 trigger 节点的 subtype。
        """
        if trigger_scores.numel() == 0:
            return torch.tensor(0.0, device=labels.device)

        # 筛选出 trigger 节点对应的 subtype
        # trigger_subtypes 是节点级 (total_nodes,),trigger_mask 选出 trigger 节点
        trigger_node_subtypes = trigger_subtypes[trigger_mask]  # (total_trigger_nodes,)

        device = trigger_scores.device
        losses = []
        B = labels.size(0)

        for b in range(B):
            label_b = labels[b].item()
            key_subtype = self.LABEL_TO_KEY_SUBTYPE.get(label_b)
            if key_subtype is None:
                continue  # clean / NullDeref 跳过

            # 该样本的 trigger 节点
            mask_b = (trigger_node_batch == b)
            if not mask_b.any():
                continue

            scores_b = trigger_scores[mask_b]
            subtypes_b = trigger_node_subtypes[mask_b]

            target_mask = (subtypes_b == key_subtype)
            other_mask = ~target_mask & (subtypes_b != 0)  # 排除 none subtype

            if not target_mask.any() or not other_mask.any():
                # 没有 target 或没有 other,无对比意义,跳过
                continue

            target_mean = scores_b[target_mask].mean()
            other_mean = scores_b[other_mask].mean()

            # margin loss:希望 target > other + margin
            loss_b = F.relu(self.margin - (target_mean - other_mean))
            losses.append(loss_b)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=False)

        return torch.stack(losses).mean()