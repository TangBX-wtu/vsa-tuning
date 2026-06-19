import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.nn import RGCNConv
from torch_geometric.utils import to_dense_batch


class CausalDualViewPool(nn.Module):
    """
    双视角池化:
      - Trigger view: trigger 节点作为 query,attention pool 全图节点
      - Sink view:    sink 节点作为 query,attention pool 全图节点

    设计:
      1. 不要求 trigger/sink 互相直接相连 —— 让 attention 自己学
      2. 没有 trigger 时,用可学习的 zero query;sink 同理
      3. 输出拼接 [trigger_view; sink_view],非对称表示
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        # 可学习的 "no trigger" / "no sink" 占位 embedding
        self.no_trigger_emb = nn.Parameter(torch.randn(dim) * 0.02)
        self.no_sink_emb = nn.Parameter(torch.randn(dim) * 0.02)

        # 新增:trigger 节点的 self-attention 权重
        self.trigger_score = nn.Linear(dim, 1)
        self.sink_score = nn.Linear(dim, 1)

        # 两个独立的 attention 头(共享会让两个视角混淆)
        self.trigger_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.sink_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=0.1, batch_first=True
        )

        # 输出投影:把双视角拼接后投回 dim 维(给 fusion 用)
        self.out_proj = nn.Linear(dim * 2, dim)

    def forward(
        self,
        node_emb: torch.Tensor,        # (total_nodes, dim)
        batch_idx: torch.Tensor,       # (total_nodes,)
        trigger_mask: torch.Tensor,    # (total_nodes,) bool
        sink_mask: torch.Tensor,       # (total_nodes,) bool
    ) -> dict:
        """
        返回 dict:
            graph_emb:       (B, dim)
            trigger_scores:  (total_trigger_nodes,) — 用于辅助损失
            trigger_node_batch: (total_trigger_nodes,) — 每个 trigger 节点属于哪个图
        """
        x_dense, node_pad_mask = to_dense_batch(node_emb, batch_idx)
        B = x_dense.size(0)
        device = x_dense.device

        # 收集所有 trigger 节点的 scores,供辅助损失使用
        all_trigger_scores = []
        all_trigger_node_batch = []

        # ── Trigger view ──────────────────────────────────────────
        trigger_query, trigger_diag = self._build_role_query_attn(
            node_emb, batch_idx, trigger_mask, B,
            self.no_trigger_emb, self.trigger_score, device,
            collect_diagnostics=True,
        )
        # trigger_diag: list of (graph_idx, scores_tensor)
        for graph_idx, scores in trigger_diag:
            all_trigger_scores.append(scores)
            all_trigger_node_batch.append(
                torch.full((scores.size(0),), graph_idx, dtype=torch.long, device=device)
            )

        kv_pad = ~node_pad_mask
        trigger_view, _ = self.trigger_attn(
            query=trigger_query, key=x_dense, value=x_dense,
            key_padding_mask=kv_pad,
        )
        trigger_view = trigger_view.squeeze(1)

        # ── Sink view(不需要诊断信息)─────────────────────────────
        sink_query, _ = self._build_role_query_attn(
            node_emb, batch_idx, sink_mask, B,
            self.no_sink_emb, self.sink_score, device,
            collect_diagnostics=False,
        )
        sink_view, _ = self.sink_attn(
            query=sink_query, key=x_dense, value=x_dense,
            key_padding_mask=kv_pad,
        )
        sink_view = sink_view.squeeze(1)

        graph_emb = self.out_proj(
            torch.cat([trigger_view, sink_view], dim=-1)
        )

        # 拼接所有 batch 的 trigger scores
        if all_trigger_scores:
            trigger_scores = torch.cat(all_trigger_scores, dim=0)
            trigger_node_batch = torch.cat(all_trigger_node_batch, dim=0)
        else:
            trigger_scores = torch.empty(0, device=device)
            trigger_node_batch = torch.empty(0, dtype=torch.long, device=device)

        return {
            "graph_emb": graph_emb,
            "trigger_scores": trigger_scores,
            "trigger_node_batch": trigger_node_batch,
        }

    @staticmethod
    def _build_role_query_attn(
            node_emb, batch_idx, role_mask, B,
            no_role_emb, score_fn, device,
            collect_diagnostics=False,
    ):
        """
        改进版:多个角色节点之间 attention pool。
        若 collect_diagnostics=True,额外返回 [(graph_idx, scores)] 列表。
        """
        D = node_emb.size(-1)
        query = torch.zeros(B, D, device=device)
        diagnostics = []

        for b in range(B):
            graph_mask = (batch_idx == b)
            role_in_graph = role_mask & graph_mask
            if role_in_graph.any():
                role_nodes = node_emb[role_in_graph]
                scores = score_fn(role_nodes).squeeze(-1)
                if collect_diagnostics:
                    diagnostics.append((b, scores))
                weights = F.softmax(scores, dim=0)
                query[b] = (role_nodes * weights.unsqueeze(-1)).sum(dim=0)
            else:
                query[b] = no_role_emb

        return query.unsqueeze(1), diagnostics


class PDGEncoder(nn.Module):
    """
    关系图卷积网络(R-GCN)+ 因果双视角池化。

    边类型(共 7 类,有向):
      0 = DDG forward  (数据依赖:定义 → 使用)
      1 = CDG forward  (控制依赖:条件 → body)
      2 = CFG forward  (控制流:源码顺序)
      3 = AST          (语法树,对称)
      4 = DDG backward
      5 = CDG backward
      6 = CFG backward

    有向边让 RGCN 能区分"沿源码顺序前进"和"逆向追溯"两种消息传递,
    从而保留代码的局部偏序信息(如 free → use vs use → free)。
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        num_layers: int,
        num_relations: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden
            out_ch = hidden if i < num_layers - 1 else out_dim
            self.convs.append(RGCNConv(in_ch, out_ch, num_relations=num_relations))

        self.res_proj = (
            nn.Linear(hidden, out_dim) if hidden != out_dim else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

        # ★ 替换原来的 global_mean_pool
        self.dual_pool = CausalDualViewPool(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        trigger_mask: torch.Tensor,    # ★ 新增
        sink_mask: torch.Tensor,        # ★ 新增
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = F.relu(h)
        res = self.res_proj(h)

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_type)
            h_new = self.dropout(F.relu(h_new))
            if i == len(self.convs) - 1:
                h_new = h_new + res
            h = h_new

        node_emb = self.norm(h)

        # 双视角池化现在返回 dict
        pool_out = self.dual_pool(node_emb, batch, trigger_mask, sink_mask)
        graph_emb = pool_out["graph_emb"]

        return node_emb, graph_emb, pool_out