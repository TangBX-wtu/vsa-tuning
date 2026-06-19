from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from transformers import AutoTokenizer

from .backbone import FrozenCodeBERT
from .gnn_encoder import PDGEncoder
from .fusion import CrossModalFusion
from .probe import VulnSemanticProbe


class VSAModel(nn.Module):
    """
    完整的 VSA-Tuning 模型。

    参数规模估计（以 CodeBERT-base 为基准）：
      - 冻结骨干：125M（不参与梯度）
      - GNN 编码器：~0.3M
      - 融合层：~2.4M
      - Probe 适配层：~0.8M
      - 分类头 + 投影头：~0.2M
      总可训练参数：约 3.7M（骨干的 3%）
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # ---- 骨干（冻结）----
        self.backbone = FrozenCodeBERT(
            config.backbone_name,
            expose_last_n=4,
        )

        # ---- 图结构编码器（可训练）----
        self.gnn = PDGEncoder(
            in_dim=128,  # line feature dim（graph_builder 输出）
            hidden=config.gnn_hidden,  # 128
            out_dim=config.gnn_hidden,  # 128
            num_layers=config.gnn_layers,
            num_relations=config.gnn_num_relations,
            dropout=config.gnn_dropout,
        )

        # ---- 跨模态融合层（可训练）----
        self.fusion = CrossModalFusion(
            seq_dim=config.backbone_hidden,  # 768
            graph_dim=config.gnn_hidden,  # 128
            out_dim=config.probe_hidden,  # 256
            num_heads=config.fusion_heads,  # 8
        )

        # ---- 漏洞语义适配层（可训练）----
        self.probe = VulnSemanticProbe(
            in_dim=config.probe_hidden,  # 256
            hidden=config.probe_hidden,  # 256
            out_dim=config.probe_hidden,  # 256
        )

        # ---- 分类头（可训练）----
        self.classifier = nn.Sequential(
            nn.Linear(config.probe_hidden + config.gnn_hidden, config.probe_hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.probe_hidden // 2, config.num_classes),
        )

        # ---- 对比投影头（可训练）----
        self.proj_head = nn.Sequential(
            nn.Linear(config.probe_hidden + config.gnn_hidden, config.probe_hidden),
            nn.GELU(),
            nn.Linear(config.probe_hidden, config.contrastive_dim),
        )

    def forward(
            self,
            input_ids: torch.Tensor,  # (B, L)
            attention_mask: torch.Tensor,  # (B, L)
            graph_batch: Batch,  # PyG Batch，包含所有图的节点/边信息
            max_graph_nodes: int = 128,  # 图节点 padding 到的最大长度
    ) -> dict:
        B = input_ids.size(0)
        device = input_ids.device

        # ---- Step 1: CodeBERT 编码（冻结，无梯度）----
        with torch.no_grad():
            bert_out = self.backbone(input_ids, attention_mask)
        seq_hidden = bert_out["last_hidden"]  # (B, L, 768)

        # ---- Step 2: GNN 编码 PDG ----
        node_emb, graph_emb, pool_out = self.gnn(
            x=graph_batch.x,
            edge_index=graph_batch.edge_index,
            edge_type=graph_batch.edge_type,
            batch=graph_batch.batch,
            trigger_mask=graph_batch.trigger_mask,  # ★ 新增
            sink_mask=graph_batch.sink_mask,  # ★ 新增
        )
        # node_emb: (total_nodes, 128)，graph_emb: (B, 128)

        # 将节点表示 pad 成 (B, max_nodes, 128) 形式，供 cross-attention 使用
        graph_node_padded, graph_pad_mask = self._pad_graph_nodes(
            node_emb, graph_batch.batch, B, max_graph_nodes, device
        )
        # graph_node_padded: (B, max_nodes, 128)
        # graph_pad_mask:    (B, max_nodes)  True=padding 节点

        # ---- Step 3: 跨模态融合 ----
        fused = self.fusion(
            seq_hidden=seq_hidden,
            graph_node_emb=graph_node_padded,
            graph_pad_mask=graph_pad_mask,
            attn_mask=attention_mask,
        )
        # fused: (B, L, 256)

        # ---- Step 4: 漏洞语义适配（Probe 注入）----
        z, attn_weights = self.probe(fused, attention_mask)
        # z:            (B, 256)  函数级漏洞语义表示
        # attn_weights: (B, L)    用于 L_probe 监督

        # ---- Step 5: 分类和对比投影 ----
        # logits = self.classifier(z)  # (B, num_classes)
        # proj_emb = F.normalize(self.proj_head(z), dim=-1)  # (B, contrastive_dim)
        # 临时验证单个样本
        '''
        if not self.training:
            with torch.no_grad():
                # attn_weights: (B, L)
                topk_vals, topk_idx = attn_weights.topk(k=10, dim=-1)
                tokenizer = AutoTokenizer.from_pretrained("/home/PythonProjects/vsa/data/CodeBert-pretrained")
                for b in range(input_ids.size(0)):
                    tokens = [tokenizer.decode([input_ids[b, i].item()])
                              for i in topk_idx[b].tolist()]
                    print(f"sample {b} top-10 attn tokens: {tokens}")
                    print(f"sample {b} top-10 attn weights: {topk_vals[b].tolist()}")
        # 临时验证
        '''

        # 将双视角图池化的信息引入分类头
        combined = torch.cat([z, graph_emb], dim=-1)  # (B, 256 + gnn_dim)
        logits = self.classifier(combined)
        proj_emb = F.normalize(self.proj_head(combined), dim=-1)

        return {
            "logits": logits,
            "proj_emb": proj_emb,
            "attn_weights": attn_weights,
            "z": z,
            "trigger_scores": pool_out["trigger_scores"],
            "trigger_node_batch": pool_out["trigger_node_batch"],
        }

    def _pad_graph_nodes(
            self,
            node_emb: torch.Tensor,  # (total_nodes, D)
            batch_idx: torch.Tensor,  # (total_nodes,)  每个节点属于哪个图
            B: int,
            max_nodes: int,
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将变长的图节点列表 pad 成固定形状 (B, max_nodes, D)。
        """
        D = node_emb.size(-1)
        padded = torch.zeros(B, max_nodes, D, device=device)
        pad_mask = torch.ones(B, max_nodes, dtype=torch.bool, device=device)

        for i in range(B):
            nodes_i = node_emb[batch_idx == i]  # (n_i, D)
            n = min(nodes_i.size(0), max_nodes)
            padded[i, :n, :] = nodes_i[:n]
            pad_mask[i, :n] = False  # False=真实节点

        return padded, pad_mask