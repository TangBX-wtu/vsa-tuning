from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VulnSemanticProbe(nn.Module):
    """
    漏洞语义适配层（Probe 注入模块）。

    核心思想（对应你方案中的 Probing/Intervention 路径）：
    不直接修改 attention 权重，而是将关键片段的表示作为一种"语义锚点"，
    通过 cross-attention 让全局表示向漏洞关键位置"对齐"。

    训练时：L_probe 监督 attention 权重与弱标注掩码 M 对齐。
    推理时：模型已学会自动聚焦到漏洞关键位置，不再需要掩码 M。
    """

    def __init__(
            self,
            in_dim: int,  # 输入维度（融合层输出维度）
            hidden: int,  # 适配层内部维度
            out_dim: int,  # 输出维度（分类头和对比头的输入维度）
    ):
        super().__init__()

        # 输入投影
        self.in_proj = nn.Linear(in_dim, hidden)

        # 漏洞语义 cross-attention：
        # Q = 全局 [CLS] 表示（代表整个函数的"问题"）
        # K/V = 所有 token 的表示（候选"回答"）
        # 输出：一个受漏洞语义引导的函数级表示
        self.vuln_attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(
            self,
            fused: torch.Tensor,  # (B, L, in_dim)  融合层输出
            attn_mask: torch.Tensor,  # (B, L)  padding mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            z:       (B, out_dim)   函数级漏洞语义表示（用于分类和对比学习）
            weights: (B, L)         每个 token 的 attention 权重（用于 L_probe 监督）
        """
        h = self.in_proj(fused)  # (B, L, hidden)

        # 用 [CLS] 的表示作为 query（形状需扩展为 (B, 1, hidden)）
        cls_query = h[:, :1, :]  # (B, 1, hidden)

        # 转换 padding mask：nn.MultiheadAttention 需要 True=忽略
        # attn_mask: 1=真实 token, 0=padding → key_padding_mask: 0=真实, 1=忽略
        key_pad_mask = (attn_mask == 0)  # (B, L)

        # Cross-attention
        attn_out, attn_weights = self.vuln_attn(
            query=cls_query,  # (B, 1, hidden)
            key=h,  # (B, L, hidden)
            value=h,  # (B, L, hidden)
            key_padding_mask=key_pad_mask,  # (B, L)
            need_weights=True,  # 返回 attention 权重用于 L_probe
            average_attn_weights=True,  # 对多头平均
        )
        # attn_out:     (B, 1, hidden)
        # attn_weights: (B, 1, L)

        attn_out = attn_out.squeeze(1)  # (B, hidden)
        weights = attn_weights.squeeze(1)  # (B, L)

        z = self.out_proj(attn_out)  # (B, out_dim)

        return z, weights