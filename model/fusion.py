import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    """
    跨模态融合层：让序列表示（来自 CodeBERT）和图表示（来自 GNN）互相调制。

    机制：
      - Query: CodeBERT 的 token 序列表示 H (B, L, 768)
      - Key/Value: GNN 的节点表示 G (B_nodes, gnn_dim)，需要先对齐到 batch

    由于 PDG 节点是行级的，而 CodeBERT token 是子词级的，
    需要一个对齐矩阵将图节点映射回 token 空间。
    这里用简单的线性投影代替精确的行-token 对齐（精确对齐见注释）。
    """

    def __init__(
            self,
            seq_dim: int,  # CodeBERT 维度，768
            graph_dim: int,  # GNN 输出维度，128
            out_dim: int,  # 融合后维度
            num_heads: int,  # cross-attention 头数
            dropout: float = 0.1,
    ):
        super().__init__()

        # 将图表示投影到与序列表示相同的维度，便于 cross-attention
        self.graph_proj = nn.Linear(graph_dim, seq_dim)

        # Multi-head cross-attention
        # Q 来自序列，K/V 来自图
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (B, L, D) 格式
        )

        # 融合后的输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(seq_dim * 2, out_dim),  # concat(原始序列, cross-attn输出)
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            seq_hidden: torch.Tensor,  # (B, L, seq_dim)  CodeBERT 最后隐层
            graph_node_emb: torch.Tensor,  # (B, max_nodes, graph_dim)  图节点表示（已 pad）
            graph_pad_mask: torch.Tensor,  # (B, max_nodes)  True 表示是 padding 节点
            attn_mask: torch.Tensor,  # (B, L)  CodeBERT attention mask
    ) -> torch.Tensor:
        """
        返回:
            fused: (B, L, out_dim)  融合后的 token 级别表示
        """
        # 将图节点表示投影到序列空间
        graph_kv = self.graph_proj(graph_node_emb)  # (B, max_nodes, seq_dim)

        # Cross-attention: token 作为 query，图节点作为 key/value
        # 每个 token 通过 attention 从程序结构中"读取"与自己相关的依赖信息
        cross_out, _ = self.cross_attn(
            query=seq_hidden,  # (B, L, seq_dim)
            key=graph_kv,  # (B, max_nodes, seq_dim)
            value=graph_kv,  # (B, max_nodes, seq_dim)
            key_padding_mask=graph_pad_mask,  # True 的位置被忽略
        )
        # cross_out: (B, L, seq_dim)

        cross_out = self.dropout(cross_out)

        # 拼接原始表示和 cross-attention 输出，再投影
        # 这保留了 CodeBERT 的功能语义，同时注入了图结构信息
        fused = self.out_proj(
            torch.cat([seq_hidden, cross_out], dim=-1)  # (B, L, seq_dim*2)
        )
        # fused: (B, L, out_dim)

        return fused