"""
三类测试：
  1. 基础性能测试（Accuracy / F1 / AUC）
  2. 泛化性测试（跨项目，检验是否脱离 Shortcut）
  3. 可解释性测试（Attention 对齐率，量化模型关注位置的合理性）
"""

import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from torch.utils.data import DataLoader

LABEL_NAMES = ["clean", "UAF", "HeapBufferOverflow", "StackBufferOverflow", "NullDeref", "IntOverflow"]


# ------------------------------------------------------------------ #
#  1. 基础性能测试
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_classification(model, loader: DataLoader, device: torch.device) -> dict:
    """
    计算多分类指标：
      - Accuracy
      - Macro F1 / Weighted F1
      - 每类的 Precision / Recall / F1
      - 混淆矩阵
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        graphs = batch["graphs"].to(device)

        outputs = model(input_ids, attn_mask, graphs)
        logits = outputs["logits"]  # (B, C)
        probs = torch.softmax(logits, dim=-1)  # (B, C)
        preds = logits.argmax(dim=-1)  # (B,)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    report = classification_report(
        labels, preds,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    # 多分类 AUC（one-vs-rest）
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")  # 某类在 batch 中未出现时可能报错

    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "auc": auc,
        "per_class": {k: report[k] for k in LABEL_NAMES if k in report},
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


# ------------------------------------------------------------------ #
#  2. 跨项目泛化测试
# ------------------------------------------------------------------ #

def cross_project_eval(model, project_loaders: dict, device: torch.device) -> dict:
    """
    跨项目测试：在项目 A 的数据上训练，在项目 B/C/D 上测试。

    project_loaders: {"linux_kernel": loader, "openssl": loader, ...}

    这是检验 Shortcut learning 的关键实验。
    如果跨项目 F1 与同项目 F1 差距 > 15%，说明模型仍然依赖项目特定特征。
    """
    results = {}
    for project_name, loader in project_loaders.items():
        metrics = evaluate_classification(model, loader, device)
        results[project_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        print("\n=== 泛化效果验证 ===")
        print(f"  [{project_name}] Acc={metrics['accuracy']:.3f}  MacroF1={metrics['macro_f1']:.3f}")
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"AUC:         {metrics['auc']:.4f}")
        print("\n每类指标：")
        for cls_name, cls_metrics in metrics["per_class"].items():
            print(f"  {cls_name:12s}  P={cls_metrics['precision']:.3f}  "
                  f"R={cls_metrics['recall']:.3f}  F1={cls_metrics['f1-score']:.3f}")
    return results


# ------------------------------------------------------------------ #
#  3. 可解释性测试：Attention 对齐率
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate_attention_alignment(
        model, loader: DataLoader, device: torch.device, top_k: int = 5
) -> dict:
    """
    量化 Probe 层 attention 权重与弱标注掩码 M 的对齐程度。

    指标：
      Precision@K：模型 attention 最高的 K 个 token 中，有多少在标注的关键片段内
      Recall@K：标注的关键片段 token 中，有多少落在模型 attention 最高的 K 个内

    这个指标是论文"可解释性分析"章节的核心数据。
    """
    model.eval()
    precisions, recalls = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        token_mask = batch["token_mask"].to(device)  # (B, L) 弱标注掩码
        graphs = batch["graphs"].to(device)

        outputs = model(input_ids, attn_mask, graphs)
        attn_weights = outputs["attn_weights"]  # (B, L)

        B = input_ids.size(0)
        for i in range(B):
            valid_len = attn_mask[i].sum().item()  # 真实 token 数量

            # 取前 K 个 attention 最高的 token 位置
            weights_i = attn_weights[i, :valid_len]  # (valid_len,)
            mask_i = token_mask[i, :valid_len]  # (valid_len,)

            # 如果这条样本没有关键片段标注（无漏洞样本），跳过
            if mask_i.sum() == 0:
                continue

            k = min(top_k, valid_len)
            topk_indices = weights_i.topk(k).indices  # 模型认为最重要的 K 个位置

            # 关键片段的 token 位置（标注值 > 0 的位置）
            key_indices = (mask_i > 0).nonzero(as_tuple=True)[0]

            # Precision@K：topk 中有多少命中了关键片段
            hits = sum(1 for idx in topk_indices if idx in key_indices)
            precisions.append(hits / k)

            # Recall@K：关键片段中有多少被 topk 覆盖
            recalls.append(hits / len(key_indices))

    return {
        f"precision@{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
    }


# ------------------------------------------------------------------ #
#  4. 消融实验辅助函数
# ------------------------------------------------------------------ #

def ablation_study(model_configs: dict, test_loader: DataLoader, device: torch.device) -> dict:
    """
    消融实验：逐个关闭损失函数组件，对比性能变化。

    model_configs: {
        "full":          (model_full,   "完整模型"),
        "no_probe":      (model_no_prob,"去掉 L_probe"),
        "no_contrastive":(model_no_con, "去掉 L_con"),
        "no_gnn":        (model_no_gnn, "去掉 GNN"),
    }
    """
    results = {}
    for config_name, (model, desc) in model_configs.items():
        metrics = evaluate_classification(model, test_loader, device)
        results[config_name] = {
            "description": desc,
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
        }
        print(f"  [{desc}] MacroF1={metrics['macro_f1']:.3f}")
    return results


# ------------------------------------------------------------------ #
#  5. 单样本推理（部署时使用）
# ------------------------------------------------------------------ #

@torch.no_grad()
def predict_one(
        model,
        code: str,
        preprocessor,  # VSAPreprocessor 实例
        device: torch.device,
) -> dict:
    """
    对单个函数代码进行推理，返回漏洞类型预测和 attention 可视化数据。
    """
    from torch_geometric.data import Batch

    model.eval()

    # 预处理
    item = {"func": code, "vuln_type": "clean"}  # vuln_type 在推理时不需要
    processed = preprocessor.process_one(item)
    if processed is None:
        return {"error": "代码过短或解析失败"}
    '''
    print(f'sanity check 临时验证边分布')
    edge_types = processed["graph"].edge_type
    print(f"Edge type distribution:")
    for t in range(7):
        count = (edge_types == t).sum().item()
        print(f"  type {t}: {count}")
    print(f"Total edges: {edge_types.size(0)}")
    '''
    input_ids = processed["input_ids"].unsqueeze(0).to(device)  # (1, L)
    attn_mask = processed["attention_mask"].unsqueeze(0).to(device)  # (1, L)
    graph_batch = Batch.from_data_list([processed["graph"]]).to(device)

    outputs = model(input_ids, attn_mask, graph_batch)

    probs = torch.softmax(outputs["logits"], dim=-1).squeeze(0)  # (C,)
    pred_cls = probs.argmax().item()
    attn_w = outputs["attn_weights"].squeeze(0)  # (L,)

    # 将 attention 权重映射回 token
    tokenizer = preprocessor.tokenizer
    tokens = tokenizer.convert_ids_to_tokens(
        processed["input_ids"].tolist()
    )

    # 取真实 token（去掉 padding）
    valid_len = processed["attention_mask"].sum().item()
    tokens = tokens[:valid_len]
    attn_valid = attn_w[:valid_len].cpu().tolist()

    return {
        "predicted_vuln_type": LABEL_NAMES[pred_cls],
        "confidence": probs[pred_cls].item(),
        "all_probs": {LABEL_NAMES[i]: probs[i].item() for i in range(len(LABEL_NAMES))},
        "attention_on_tokens": list(zip(tokens, attn_valid)),
        # 前 10 个最高 attention 的 token，方便快速定位
        "top_attention_tokens": sorted(
            zip(tokens, attn_valid), key=lambda x: x[1], reverse=True
        )[:10],
    }


# ------------------------------------------------------------------ #
#  5. 对已经进行预处理的单样本推理（测试时使用）
# ------------------------------------------------------------------ #

@torch.no_grad()
def predict_one_pt(
        model,
        processed,
        device: torch.device,
) -> dict:
    """
    对已经经过预处理的单个函数代码进行推理，返回漏洞类型预测。
    """
    from torch_geometric.data import Batch
    model.eval()

    input_ids = processed["input_ids"].unsqueeze(0).to(device)  # (1, L)
    attn_mask = processed["attention_mask"].unsqueeze(0).to(device)  # (1, L)
    graph_batch = Batch.from_data_list([processed["graph"]]).to(device)

    outputs = model(input_ids, attn_mask, graph_batch)

    probs = torch.softmax(outputs["logits"], dim=-1).squeeze(0)  # (C,)
    pred_cls = probs.argmax().item()

    # print(f'test: pred_cls is {pred_cls}')
    return {
        "predicted_vuln_type": LABEL_NAMES[pred_cls],
        "confidence": probs[pred_cls].item(),
        "all_probs": {LABEL_NAMES[i]: probs[i].item() for i in range(len(LABEL_NAMES))},
    }
