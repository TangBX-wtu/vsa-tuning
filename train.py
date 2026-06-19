import time
import torch
import sys
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as GeomLoader

from config import VSAConfig
from model.vsa_model import VSAModel
from loss.losses import VSALoss
from data.dataset import build_dataloader
from data import preprocess
sys.modules['preprocess'] = preprocess


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch):
    model.train()
    total_losses = {"total": 0, "cls": 0, "probe": 0, "contrastive": 0, "focus": 0}
    n_batches = len(loader)

    for batch_idx, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        key_mask = batch["token_mask"].to(device)  # (B, L) 弱标注掩码
        graph_batch = batch["graphs"].to(device)  # PyG Batch

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            graph_batch=graph_batch,
        )

        # 计算联合损失
        losses = loss_fn(
            logits=outputs["logits"],
            labels=labels,
            attn_weights=outputs["attn_weights"],
            key_mask=key_mask,
            padding_mask=attn_mask,
            proj_emb=outputs["proj_emb"],
            trigger_scores=outputs["trigger_scores"],  # ★ 新增
            trigger_node_batch=outputs["trigger_node_batch"],  # ★ 新增
            trigger_subtypes=graph_batch.trigger_subtype,  # ★ 新增
            trigger_mask_nodes=graph_batch.trigger_mask,  # ★ 新增
        )

        # print(f"[train诊断] labels传入损失函数: {labels.tolist()}, dtype={labels.dtype}")

        optimizer.zero_grad()
        losses["total"].backward()

        # 梯度裁剪，防止不稳定（适配层参数较少时梯度可能较大）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()

        # 打印过程信息

        log_every = max(1, n_batches // 10)
        if batch_idx % log_every == 0 or batch_idx == n_batches:
            unique_labels = labels.unique().tolist()
            print(
                f"  Epoch {epoch:02d} [{batch_idx:4d}/{n_batches}] "
                f"loss={losses['total'].item():.4f} "
                f"cls={losses['cls'].item():.3f} "
                f"probe={losses['probe'].item():.3f} "
                f"con={losses['contrastive'].item():.3f} "
                f"| batch_classes={unique_labels}"
                f"focus={losses['focus']:.3f} "
            )

    n = len(loader)
    return {k: v / n for k, v in total_losses.items()}


def validate(model, loader, loss_fn, device):
    """验证集评估，用于判断是否保存 checkpoint"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attention_mask"].to(device)
            labels     = batch["labels"].to(device)
            key_mask   = batch["token_mask"].to(device)
            graphs     = batch["graphs"].to(device)

            outputs = model(input_ids, attn_mask, graphs)
            losses  = loss_fn(
                logits=outputs["logits"],
                labels=labels,
                attn_weights=outputs["attn_weights"],
                key_mask=key_mask,
                padding_mask=attn_mask,
                proj_emb=outputs["proj_emb"],
                trigger_scores=outputs["trigger_scores"],  # ★ 新增
                trigger_node_batch=outputs["trigger_node_batch"],  # ★ 新增
                trigger_subtypes=graphs.trigger_subtype,  # ★ 新增
                trigger_mask_nodes=graphs.trigger_mask,  # ★ 新增
            )
            total_loss += losses["total"].item()
            correct += (outputs["logits"].argmax(-1) == labels).sum().item()
            total   += labels.size(0)

    return {"val_loss": total_loss / len(loader), "val_acc": correct / total}


def main():
    config = VSAConfig()
    config.print_config_arr()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    train_loader = build_dataloader("data/processed/MIXED/TITAN_PRIME_DIVERSEVUL_train_v6.pt", config.batch_size, shuffle=True)
    val_loader = build_dataloader("data/processed/MIXED/TITAN_PRIME_DIVERSEVUL_val_v6.pt",   config.batch_size, shuffle=False)
    print(f"训练集: {len(train_loader.dataset)} 条  验证集: {len(val_loader.dataset)} 条")

    # 构建模型
    model    = VSAModel(config).to(device)
    loss_fn  = VSALoss(config)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"可训练参数: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=config.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        # warmup:前 2 epoch 不开 focus loss
        if epoch <= 2:
            loss_fn.epsilon = 0.0
        elif epoch <= 5:
            loss_fn.epsilon = 0.15
        else:
            loss_fn.epsilon = config.epsilon

        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(cls={train_metrics['cls']:.3f} "
            f"probe={train_metrics['probe']:.3f} "
            f"con={train_metrics['contrastive']:.3f}) | "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"val_acc={val_metrics['val_acc']:.3f}"
        )

        # 保存最优 checkpoint
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "config":      config,
                "val_loss":    best_val_loss,
            }, "checkpoints/best_model.pt")
            print(f"  -> 保存 checkpoint（val_loss={best_val_loss:.4f}）")


def redirect_log(redirect: bool, task_name: str):
    if redirect:
        current_time = time.asctime()
        log_file = './Logs/' + task_name + '_' + str(current_time) + '.txt'
        sys.stdout = open(log_file, 'w')


if __name__ == "__main__":
    redirect_log(True, "TITAN_PRIME_DIVERSEVUL_MIXED_v6")
    print("---基于PRIME数据集+TITAN+DIVERSE混合V6---")
    main()
