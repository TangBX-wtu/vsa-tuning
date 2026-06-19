from dataclasses import dataclass


@dataclass
class VSAConfig:
    # 骨干模型
    backbone_name: str = "./data/CodeBert-pretrained"
    backbone_hidden: int = 768
    freeze_backbone: bool = True

    # 图编码器
    gnn_hidden: int = 128
    gnn_layers: int = 3
    gnn_num_relations: int = 7  # 有序 DDG + CDG + AST
    gnn_dropout: float = 0.1

    # 融合与适配层
    fusion_heads: int = 8
    probe_hidden: int = 256

    # 分类
    num_classes: int = 6  # 0:无漏洞 1.UAF 2.HeapBufferOverflow 3.StackBufferOverflow 4.NullDeref 5.IntOverflow
    contrastive_dim: int = 128  # 对比学习投影维度

    # 训练
    max_seq_len: int = 512
    batch_size: int = 32
    lr: float = 2e-4  # 只训练适配层，lr 可以比全量微调大
    epochs: int = 30
    warmup_steps: int = 500

    # 损失权重（消融实验时逐个置零）
    alpha: float = 1.0  # L_cls 权重
    beta: float = 0.3  # L_probe 权重
    gamma: float = 0.5  # L_con 权重
    temperature: float = 0.07  # 对比损失温度
    probe_temp: float = 0.5  # Probe 标签温度
    epsilon = 0.3  # trigger focus loss 权重，建议范围0.1-0.5，太高会过度压制 attention pool 的灵活性,太低没效果

    def print_config_arr(self):
        print("---- 骨干模型参数 ----")
        print(f"backbone_name: {self.backbone_name}")
        print(f"backbone_hidden : {self.backbone_hidden}")
        print(f"freeze_backbone : {self.freeze_backbone}")
        print("\n")
        print("---- 图编码器参数 ----")
        print(f"gnn_hidden : {self.gnn_hidden}")
        print(f"gnn_layers : {self.gnn_layers}")
        print(f"gnn_num_relations : {self.gnn_num_relations}")
        print(f"gnn_dropout : {self.gnn_dropout}")
        print("\n")
        print("---- 融合与适配层参数 ----")
        print(f"fusion_heads : {self.fusion_heads}")
        print(f"probe_hidden : {self.probe_hidden}")
        print("\n")
        print("---- 分类器参数 ----")
        print(f"num_classes : {self.num_classes}; 检测类别: 0:无漏洞 1.UAF 2.HeapBufferOverflow 3.StackBufferOverflow "
              f"4.OSCommandInjection 5.LDAPInjection 6.NullDeref 7.IntOverflow")
        print(f"contrastive_dim : {self.contrastive_dim}")
        print("\n")
        print("---- 训练参数 ----")
        print(f"max_seq_len : {self.max_seq_len}")
        print(f"batch_size : {self.batch_size}")
        print(f"lr : {self.lr}")
        print(f"epochs : {self.epochs}")
        print(f"warmup_steps : {self.warmup_steps}")
        print("\n")
        print("---- 损失权重参数 ----")
        print(f"alpha(L_cls 权重) : {self.alpha}")
        print(f"beta(L_probe 权重) : {self.beta}")
        print(f"gamma(L_con 权重) : {self.gamma}")
        print(f"epsilon(L_foucus 权重): {self.epsilon}")
        print(f"temperature(对比损失温度) : {self.temperature}")
        print(f"probe_temp(Probe 软标签温度) : {self.probe_temp}")
