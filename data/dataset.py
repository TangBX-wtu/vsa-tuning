"""
PyTorch Dataset：加载预处理好的 .pt 文件，提供 collate_fn。

新增 BalancedBatchSampler：
  每个 batch 中保证每个类别至少出现 min_samples_per_class 个样本，
  从根本上解决对比损失"孤立类别"问题。
"""
from typing import List, Iterator

import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from torch_geometric.data import Batch


class VSADataset(Dataset):
    def __init__(self, pt_path: str):
        self.samples = torch.load(pt_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BalancedBatchSampler(Sampler):
    """
    每个 batch 保证每个类别至少出现 min_per_class 个样本。

    策略：
      1. 按类别建立索引桶。
      2. 每次构造 batch 时，先从每个类别各取 min_per_class 个，
         再随机补齐到 batch_size。
      3. 若某类别样本不足 min_per_class，则有放回地重复采样。
    """

    def __init__(
        self,
        dataset: VSADataset,
        batch_size: int,
        min_per_class: int = 2,
        drop_last: bool = True,
    ):
        super().__init__(dataset)
        self.batch_size    = batch_size
        self.min_per_class = min_per_class
        self.drop_last     = drop_last

        # 按标签建桶
        self.label_to_indices: dict[int, List[int]] = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            label = int(sample["label"].item())
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())
        self.n_classes = len(self.labels)

        # 需要的最少 batch 占位数
        self._min_slots = self.n_classes * min_per_class
        assert self._min_slots <= batch_size, (
            f"batch_size ({batch_size}) 太小，无法容纳 "
            f"{self.n_classes} 个类别各 {min_per_class} 个样本 "
            f"（需要至少 {self._min_slots}）"
        )

        # 总样本数（用于估算 epoch 长度）
        self.total = len(dataset)

    def __iter__(self) -> Iterator[List[int]]:
        # 每个 epoch 开始时，拷贝并打乱各桶
        buckets = {
            lbl: idxs.copy()
            for lbl, idxs in self.label_to_indices.items()
        }
        for idxs in buckets.values():
            random.shuffle(idxs)

        # 桶指针
        pointers = {lbl: 0 for lbl in self.labels}
        all_indices = list(range(self.total))
        random.shuffle(all_indices)
        global_ptr = 0

        num_batches = self.total // self.batch_size

        for _ in range(num_batches):
            batch: List[int] = []

            # 每个类别各取 min_per_class 个（有放回）
            for lbl in self.labels:
                idxs = buckets[lbl]
                for _ in range(self.min_per_class):
                    ptr = pointers[lbl] % len(idxs)
                    batch.append(idxs[ptr])
                    pointers[lbl] += 1

            # 补齐剩余位置（随机全局采样，允许重复）
            remaining = self.batch_size - len(batch)
            while remaining > 0:
                batch.append(all_indices[global_ptr % len(all_indices)])
                global_ptr += 1
                remaining -= 1

            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.total // self.batch_size


def collate_fn(batch: List[dict]) -> dict:
    """将一个 batch 的样本合并。"""
    return {
        "input_ids":      torch.stack([s["input_ids"] for s in batch]),       # (B, L)
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),  # (B, L)
        "token_mask":     torch.stack([s["token_mask"] for s in batch]),      # (B, L)
        "labels":         torch.stack([s["label"] for s in batch]),           # (B,)
        "graphs":         Batch.from_data_list([s["graph"] for s in batch]),  # PyG Batch
        "vuln_types":     [s["vuln_type"] for s in batch],                    # list of str
    }


def build_dataloader(
    pt_path: str,
    batch_size: int,
    shuffle: bool = True,
    min_per_class: int = 2,
) -> DataLoader:
    dataset = VSADataset(pt_path)

    if shuffle:
        # 训练集：使用 BalancedBatchSampler 保证对比学习的正样本对
        sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            min_per_class=min_per_class,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,   # 注意：用 batch_sampler 而非 batch_size+sampler
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
    else:
        # 验证集：顺序加载即可
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )
