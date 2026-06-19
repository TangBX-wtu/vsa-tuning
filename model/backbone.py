import torch
import torch.nn as nn
from transformers import AutoModel


class FrozenCodeBERT(nn.Module):
    """
    冻结 CodeBERT 的所有参数，只暴露最后几层的隐层表示。
    保留最后 N 层用于后续融合（默认 N=4）。
    """

    def __init__(self, model_name: str, expose_last_n: int = 4):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.expose_last_n = expose_last_n

        # 冻结所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(
            self,
            input_ids: torch.Tensor,  # (B, L)
            attention_mask: torch.Tensor,  # (B, L)
    ) -> dict:
        """
        返回:
            last_hidden: (B, L, 768)  最后一层隐层，用于 Probe 注入
            pooled:      (B, 768)     [CLS] 表示，用于分类
            all_hidden:  list of (B, L, 768)  最后 N 层，用于融合
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # outputs.hidden_states: tuple of (num_layers+1) tensors, each (B, L, 768)
        hidden_states = outputs.hidden_states  # 包含 embedding 层 + 12 层

        # 取最后 N 层
        last_n_hidden = hidden_states[-self.expose_last_n:]  # list of (B, L, 768)

        # 最后一层
        last_hidden = hidden_states[-1]  # (B, L, 768)

        # [CLS] 的表示（第 0 个 token）
        pooled = last_hidden[:, 0, :]  # (B, 768)

        return {
            "last_hidden": last_hidden,
            "pooled": pooled,
            "last_n_hidden": last_n_hidden,
        }