"""
数据预处理主流程。
输入：原始 JSON 数据集（见下方格式说明）
输出：处理好的 .pt 文件，可直接被 DataLoader 加载

Joern 集成说明（Windows）：
  需将 Joern 安装根目录配置到环境变量 JOERN_HOME，或在实例化时通过
  joern_home 参数显式指定。可执行文件路径为 <JOERN_HOME>/joern.bat。
  若 Joern 不可用或单次解析超时/失败，自动 fallback 到规则化 PDG 构建。
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import tempfile
import platform
from collections import Counter

from pathlib import Path
import random
from typing import List, Dict, Tuple

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer
from tqdm import tqdm

# 删除这两类样本，因为样本会误导模型
# "OSCommandInjection": 4,  # CWE-78
# "LDAPInjection": 5,  # CWE-90
LABEL_MAP: Dict[str, int] = {
    "clean": 0,
    "UAF": 1,  # CWE-416
    "HeapBufferOverflow": 2,  # CWE-122
    "StackBufferOverflow": 3,  # CWE-121
    "NullDeref": 4,  # CWE-476
    "IntOverflow": 5,  # CWE-190
}

# 显式剔除数据集中废弃的漏洞类型，防止污染clean类别
DEPRECATED_TYPES = {"OSCommandInjection", "LDAPInjection"}

# ─────────────────────────────────────────────────────────────────────────────
#  弱监督标注规则
#  每个 entry 包含：
#    trigger_keywords  — 直接触发该漏洞的 API / 操作，命中即标注为关键行
#    context_keywords  — 在触发行附近出现时加强标注的上下文特征
#    param_indicators  — 函数签名中表明外部输入的参数名片段（用于签名行标注）
# ─────────────────────────────────────────────────────────────────────────────
VULN_PATTERNS: Dict[str, dict] = {

    # ── UAF (CWE-416) ────────────────────────────────────────────────────────
    # 关键操作:释放 → 之后的解引用/打印
    "UAF": {
        "trigger_keywords": [
            # 通用
            "free(", "delete ", "delete[",
            # Linux kernel
            "kfree(", "vfree(", "kvfree(",
            "kfree_skb(", "kfree_rcu(",  # SKB 释放,常见 UAF 来源
            "kmem_cache_free(", "kmem_cache_destroy(",
            "put_device(", "put_disk(", "put_page(",
            "iput(", "dput(",  # inode/dentry 释放
            # GLib / GObject
            "g_free(", "g_object_unref(", "g_clear_object(",
            "g_slice_free(",
            # OpenSSL
            "OPENSSL_free(", "EVP_PKEY_free(", "X509_free(",
            "BN_free(", "BIO_free(",
            # 引用计数 put 类(常见 UAF 模式)
            "refcount_dec_and_test(", "atomic_dec_and_test(",
        ],
        "context_keywords": [
            "->",
            # 通用 use sink
            "printf", "fprintf", "snprintf", "sprintf",
            "memcpy", "memmove", "strcpy",
            # SARD 特有(混合训练时仍保留)
            "printIntLine", "printLongLongLine", "printLine",
            "printWLine", "printHexCharLine",
        ],
    },

    # ── NullDeref (CWE-476) ──────────────────────────────────────────────────
    # 关键操作:对可能为 NULL 的指针解引用
    # 注意:单字符 * 太歧义,移除;靠 -> 和 NULL 比较 + 函数签名指针参数
    "NullDeref": {
        "trigger_keywords": [
            "->",
            "IS_ERR(",  # Linux kernel 错误指针检查模式
            "PTR_ERR(",
        ],
        "context_keywords": [
            "NULL", "nullptr",
            # Linux kernel 常见 alloc(返回可能 NULL)
            "kmalloc(", "kzalloc(", "kcalloc(", "vmalloc(",
            "alloc_skb(", "kmem_cache_alloc(",
            # 通用
            "malloc(", "calloc(", "realloc(",
            # 函数返回指针的常见形式(可能为 NULL 但没检查)
            "find_", "lookup_", "get_", "alloc_",
        ],
        "param_indicators": ["*", "**"],
    },

    # ── IntOverflow (CWE-190) ────────────────────────────────────────────────
    # 关键操作:整数运算后用作 size/length
    # 单字符 + - * 歧义太大,改为含语义的多字符词
    "IntOverflow": {
        "trigger_keywords": [
            # 输入转换
            "atoi(", "atol(", "atoll(",
            "strtol(", "strtoul(", "strtoll(", "strtoull(",
            "kstrtoul(", "kstrtoint(",  # kernel 版本
            "simple_strtoul(",  # Linux kernel
            # 用作 size 的危险位置
            "malloc(", "calloc(", "realloc(",
            "kmalloc(", "kzalloc(", "kvmalloc(",
            "memcpy(", "memmove(", "memset(",
            "alloca(", "vmalloc(",
            # 用户空间数据拷贝(Linux kernel,size 来源不可信)
            "copy_from_user(", "copy_to_user(",
            "get_user(", "put_user(",
        ],
        "context_keywords": [
            "+=", "-=", "*=", "<<=",
            "INT_MAX", "INT_MIN", "SIZE_MAX", "ULONG_MAX",
            "U32_MAX", "U64_MAX",  # kernel 宏
            "overflow", "wrap",  # 注释或变量名
        ],
    },

    # ── HeapBufferOverflow (CWE-122) ─────────────────────────────────────────
    # 关键操作:堆缓冲区上的危险拷贝/写入
    "HeapBufferOverflow": {
        "trigger_keywords": [
            "memcpy(", "memmove(", "memset(",
            "strcpy(", "strcat(", "sprintf(", "snprintf(",
            "strncpy(", "strncat(",
            "gets(", "scanf(", "fscanf(", "fgets(",
            "read(", "fread(", "recv(", "recvfrom(",
            # Linux kernel
            "copy_from_user(", "copy_to_user(",
            "skb_put(", "skb_pull(",  # SKB 操作,常见溢出点
            "__builtin_memcpy(", "__builtin_strcpy(",  # 编译器内建
        ],
        "context_keywords": [
            # 堆分配(确认是堆上缓冲区)
            "malloc(", "calloc(", "realloc(",
            "kmalloc(", "kzalloc(", "kvmalloc(",
            "alloc_skb(",
            "new ", "new[",
        ],
        "param_indicators": ["len", "size", "length", "sz", "count"],
    },

    # ── StackBufferOverflow (CWE-121) ────────────────────────────────────────
    # 关键操作:栈数组上的危险拷贝/写入
    # 触发函数与 HBO 重叠,context 改为栈数组类型声明的语义关键词
    "StackBufferOverflow": {
        "trigger_keywords": [
            "memcpy(", "memmove(", "memset(",
            "strcpy(", "strcat(", "sprintf(", "snprintf(",
            "strncpy(", "strncat(",
            "gets(", "scanf(", "fscanf(", "fgets(",
            "read(", "fread(", "recv(", "recvfrom(",
            # Linux kernel
            "copy_from_user(", "copy_to_user(",
            "skb_put(", "skb_pull(",  # SKB 操作,常见溢出点
            "__builtin_memcpy(", "__builtin_strcpy(",
        ],
        "context_keywords": [
            # 栈数组类型(BPE 后大概率作为单 token,避免 char/int 这类太通用的词)
            "uint8_t", "uint16_t", "uint32_t",
            "int8_t", "int16_t", "int32_t",
            "BYTE", "WORD", "DWORD",
            # 注意:char 和 int 太通用,会把所有变量声明行都拉进来,这里不放
        ],
        "param_indicators": ["len", "size", "length", "sz", "count"],
    },

    # ── OSCommandInjection (CWE-78) ──────────────────────────────────────────
    # 关键操作:外部输入 → 命令执行
    "OSCommandInjection": {
        "trigger_keywords": [
            "system(", "popen(", "_popen(",
            "execve(", "execvp(", "execvpe(", "execl(", "execlp(",
            "execve_user(",  # kernel
            "ShellExecute(", "CreateProcess(", "WinExec(",
            "wordexp(",
            # PHP/scripting wrapper(如果 PRIMEVUL 包含 PHP 代码)
            "shell_exec(", "passthru(",
        ],
        "context_keywords": [
            # 外部输入来源
            "argv", "getenv(",
            "scanf(", "gets(", "fgets(", "fscanf(",
            "recv(", "read(", "fread(",
            # 不安全字符串拼接(连接输入到命令)
            "snprintf(", "sprintf(", "strcat(", "strcpy(",
        ],
        "param_indicators": ["cmd", "command", "input", "data"],
    },

    # ── LDAPInjection (CWE-90) ───────────────────────────────────────────────
    # 关键操作:外部输入 → LDAP 查询
    "LDAPInjection": {
        "trigger_keywords": [
            "ldap_search(", "ldap_search_s(",
            "ldap_search_ext(", "ldap_search_ext_s(",
            "ldap_modify(", "ldap_add(", "ldap_delete(",
            "ldap_bind(", "ldap_simple_bind(",
        ],
        "context_keywords": [
            # 外部输入来源
            "argv", "getenv(",
            "recv(", "read(", "fgets(",
            # 注入点(查询字符串拼接)
            "sprintf(", "snprintf(", "strcat(", "strcpy(",
            # LDAP 语义词(辅助识别)
            "filter", "ldap_dn",
        ],
        "param_indicators": ["filter", "query", "input", "data"],
    },
}

# 节点特征关键词表（模块级常量，避免每次重建）
# 相比原始版本扩充了 injection 相关词
_NODE_KEYWORDS: List[str] = [
    "free", "malloc", "delete", "new", "return", "goto",
    "if", "else", "while", "for", "null", "nullptr", "NULL",
    "->", "*(", "++", "--", "<<", ">>", "sizeof", "memcpy",
    "strcpy", "sprintf", "gets", "scanf", "read", "write",
    "open", "close", "lock", "unlock", "assert", "abort",
    "exit", "throw", "catch", "try", "realloc", "calloc",
    "kfree", "kmalloc", "vmalloc", "kzalloc", "devm_",
    "pci_", "usb_", "net_", "sk_buff", "struct ", "typedef",
    "int ", "char ", "void ", "unsigned ", "size_t", "uint",
    "ptr", "buf", "len", "size", "count", "num", "idx",
    "err", "ret", "status", "flag", "mask", "lock", "mutex",
    # 新增：injection 类漏洞相关
    "system(", "popen(", "execve(", "ldap_search", "ldap_bind",
    "filter", "command", "query", "argv",
]
_KW_DIM = 64  # one-hot 维度（只取前 64 个关键词）


# ─────────────────────────────────────────────────────────────────────────────
#  从 TRIGGER_KEYWORDS / SINK_KEYWORDS 派生出函数名集合和操作符列表
#  在节点角色识别时分别处理,避免 AST 嵌套节点被子串误匹配
# ─────────────────────────────────────────────────────────────────────────────

def _split_keywords(keywords: List[str]) -> Tuple[set, List[str]]:
    """
    把混合的 keyword 列表拆分为:
      - func_names: 以 '(' 结尾的,提取函数名 → set
      - ops:        其他(操作符等)→ list
    """
    func_names = set()
    ops = []
    for kw in keywords:
        kw_stripped = kw.strip()
        if not kw_stripped:
            continue
        if kw_stripped.endswith("("):
            func_names.add(kw_stripped.rstrip("(").strip())
        elif kw_stripped.endswith(" "):  # 'new ' 'delete ' 这种
            func_names.add(kw_stripped.strip())
        else:
            ops.append(kw_stripped)
    return func_names, ops


# ─────────────────────────────────────────────────────────────────────────────
#  节点角色识别(用于双视角图池化)
#    TRIGGER: dangerous API 调用,因果链的"因"
#    SINK:    疑似受影响位置,因果链的"果"
#  注意:这是跨漏洞类型的"通用"角色识别,不区分具体 CWE。
#  推理时模型自己学习"哪种 trigger × 哪种 sink"对应哪种漏洞。
#  TDB:需要扩展PRIMEVUL，CLEANVUL和TITANVUL的字段
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_KEYWORDS: List[str] = [
    # 内存释放
    "free", "kfree", "vfree", "g_free", "delete",
    # 内存分配
    "malloc", "calloc", "realloc", "kmalloc", "kzalloc", "vmalloc",
    # 危险拷贝
    "memcpy", "memmove", "memset", "strcpy", "strcat",
    "sprintf", "snprintf", "strncpy", "strncat",
    # 输入读取
    "gets", "scanf", "fscanf", "fgets", "read", "fread", "recv", "recvfrom",
    # 命令执行
    "system", "popen", "execve", "execl", "execlp", "execvp",
    "ShellExecute", "CreateProcess", "WinExec",
    # LDAP
    "ldap_search", "ldap_search_s", "ldap_modify", "ldap_bind",
    # 整数转换
    "atoi", "atol", "strtol", "strtoul",
]

SINK_KEYWORDS: List[str] = [
    # 解引用形式
    "->",
    # SARD-specific use sink
    "printIntLine", "printLongLongLine", "printLine",
    "printWLine", "printHexCharLine",
    # 通用打印
    "printf", "fprintf",
    # 写入操作(可能是受 buffer 影响)
    "memcpy", "memmove", "strcpy", "sprintf",
]

# TRIGGER_KEYWORDS 全是函数名,直接转 set
_TRIGGER_FUNCS = {kw.strip().rstrip("(") for kw in TRIGGER_KEYWORDS if kw.strip()}
_SINK_FUNCS_FROM_LIST = {kw.strip().rstrip("(") for kw in SINK_KEYWORDS if kw.strip()}

# SINK_KEYWORDS 里 '->' 是操作符,要单独挑出来
_SINK_OPS = [kw for kw in SINK_KEYWORDS if not kw.strip()[0:1].isalpha()]
_SINK_FUNCS = _SINK_FUNCS_FROM_LIST - set(_SINK_OPS)

# 节点 code 长度阈值:超过此长度的节点(通常是 BLOCK / 整段函数体)
# 不参与角色识别,避免误激活
_NODE_CODE_MAX_LEN = 100

# ─────────────────────────────────────────────────────────────────────────────
#  Trigger 子类型(用于辅助监督)
#  每个 vuln_type 在结构上由某个 subtype 的 trigger 节点主导:
#    UAF  → 释放型(free 系列)
#    HBO/SBO → 拷贝型(memcpy 系列)
#    IntOverflow → 分配/拷贝型
#  辅助损失会强制让"关键 subtype"的 trigger 节点拿到更高的池化权重
# ─────────────────────────────────────────────────────────────────────────────

# subtype 编码(int,便于在 Data 里存为 tensor)
SUBTYPE_TO_ID: Dict[str, int] = {
    "none": 0,
    "free": 1,
    "alloc": 2,
    "copy": 3,
    "input": 4,
    "exec": 5,
}

TRIGGER_SUBTYPE_MAP: Dict[str, List[str]] = {
    "free": [
        "free", "kfree", "vfree", "kvfree",
        "delete",
        "g_free", "g_object_unref", "g_clear_object", "g_slice_free",
        "OPENSSL_free", "EVP_PKEY_free", "X509_free", "BN_free", "BIO_free",
        "kfree_skb", "kfree_rcu", "kmem_cache_free",
        "put_device", "put_disk", "put_page",
    ],
    "alloc": [
        "malloc", "calloc", "realloc",
        "kmalloc", "kzalloc", "kcalloc",
        "vmalloc", "kvmalloc",
        "alloc_skb", "kmem_cache_alloc",
        "new",
    ],
    "copy": [
        "memcpy", "memmove", "memset",
        "strcpy", "strcat", "strncpy", "strncat",
        "sprintf", "snprintf",
        "__builtin_memcpy", "__builtin_strcpy",
        "copy_from_user", "copy_to_user",
    ],
    "input": [
        "scanf", "fscanf", "gets", "fgets",
        "read", "fread", "recv", "recvfrom",
        "get_user",
    ],
    "exec": [
        "system", "popen", "_popen",
        "execve", "execl", "execlp", "execvp", "execvpe",
        "ShellExecute", "CreateProcess", "WinExec",
    ],
}

# 反向查找:funcname → subtype
_FUNC_TO_SUBTYPE: Dict[str, str] = {}
for subtype, funcs in TRIGGER_SUBTYPE_MAP.items():
    for f in funcs:
        _FUNC_TO_SUBTYPE[f] = subtype


def _classify_trigger_subtype(code: str) -> int:
    """
    返回 trigger 子类型的 ID(int)。非 trigger 节点返回 SUBTYPE_TO_ID['none']=0。
    沿用 _classify_node_role 的精确匹配策略。
    """
    if not code:
        return SUBTYPE_TO_ID["none"]
    code_stripped = code.strip()
    if not code_stripped:
        return SUBTYPE_TO_ID["none"]

    outer_call = re.match(r'^(\w+)\s*\(', code_stripped)
    if outer_call:
        fname = outer_call.group(1)
        subtype = _FUNC_TO_SUBTYPE.get(fname)
        if subtype is not None:
            return SUBTYPE_TO_ID[subtype]
    return SUBTYPE_TO_ID["none"]


class VSAGraphData(Data):
    """
    自定义 Data 类,声明 trigger_mask 和 sink_mask 是节点级属性,
    让 Batch.from_data_list 正确合并它们。
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        # 节点级 mask 沿 dim 0 拼接(和 num_nodes 同步)
        if key in ("trigger_mask", "sink_mask", "key_mask", "trigger_subtype"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        # 这些 mask 不是索引,合并时不需要加偏移
        if key in ("trigger_mask", "sink_mask", "key_mask", "trigger_subtype"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def _classify_node_role(code: str) -> Tuple[bool, bool]:
    """
    精确匹配版本(修复 AST 嵌套节点误激活):
      - 函数调用类:仅当节点 code 的最外层是已知 trigger/sink 函数时命中
      - 操作符类(如 '->'):子串匹配 + 长度限制
    """
    if not code:
        return False, False

    code_stripped = code.strip()
    if not code_stripped:
        return False, False

    is_trigger = False
    is_sink = False

    # ── 函数调用类:正则匹配最外层 funcname( 形式 ────────────────────
    outer_call = re.match(r'^(\w+)\s*\(', code_stripped)
    if outer_call:
        func_name = outer_call.group(1)
        if func_name in _TRIGGER_FUNCS:
            is_trigger = True
        if func_name in _SINK_FUNCS:
            is_sink = True

    # ── 操作符类 sink(只有 '->'):仅在节点不是 BLOCK 时考虑 ────────────
    if len(code_stripped) < _NODE_CODE_MAX_LEN and not is_sink:
        is_sink = any(op in code_stripped for op in _SINK_OPS)

    # ── 注意:trigger 没有"操作符类",所有 TRIGGER_KEYWORDS 都是函数名 ──
    # 不再走子串匹配分支,杜绝 AST 嵌套节点的误命中

    return is_trigger, is_sink


# ─────────────────────────────────────────────────────────────────────────────
#  Joern CSV 解析工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _parse_joern_nodes(
        nodes_csv: Path,
) -> Tuple[Dict[int, int], List[str], List[int]]:
    """
    解析 Joern 导出的 nodes.csv（Tab 分隔）。

    返回：
      id_to_idx   — {joern_node_id: 本地从 0 开始的节点索引}
      node_codes  — 每个节点对应的代码片段（用于特征提取）
      node_lines  — 每个节点对应的源码行号（1-indexed，-1 表示未知）
    """
    # 只保留语句级节点，过滤 METHOD / FILE 等元节点
    KEEP_LABELS = {
        "CALL", "IDENTIFIER", "LITERAL",
        "METHOD_PARAMETER_IN", "METHOD_RETURN",
        "CONTROL_STRUCTURE", "BLOCK", "LOCAL", "RETURN",
    }
    id_to_idx: Dict[int, int] = {}
    node_codes: List[str] = []
    node_lines: List[int] = []

    with open(nodes_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("_label", "") not in KEEP_LABELS:
                continue
            try:
                nid = int(row["id"])
            except (KeyError, ValueError):
                continue
            code = (row.get("code") or row.get("name") or "").replace("\t", " ")
            try:
                lineno = int(row.get("lineNumber", -1))
            except ValueError:
                lineno = -1
            id_to_idx[nid] = len(node_codes)
            node_codes.append(code)
            node_lines.append(lineno)

    return id_to_idx, node_codes, node_lines


def _parse_joern_edges(
        edges_csv: Path,
        id_to_idx: Dict[int, int],
) -> Tuple[List[int], List[int], List[int]]:
    """
    解析 Joern 导出的 edges.csv(Tab 分隔)。

    边类型编码(共 7 类,有向):
      0 = DDG forward  (数据依赖正向:定义 → 使用)
      1 = CDG forward  (控制依赖正向:条件 → body)
      2 = CFG forward  (控制流正向:源码顺序)
      3 = AST          (语法树,对称,不加反向边)
      4 = DDG backward
      5 = CDG backward
      6 = CFG backward

    返回 (src_list, dst_list, edge_type_list)。
    """
    EDGE_TYPE_MAP_DIRECTED: Dict[str, int] = {
        "DATA_DEP": 0,  # DDG forward
        "REACHING_DEF": 0,
        "CDG": 1,  # CDG forward
        "CONTROL_DEP": 1,
        "CFG": 2,  # CFG forward (源码顺序,这是关键)
        "AST": 3,  # AST (对称,不加反向)
    }

    src_list: List[int] = []
    dst_list: List[int] = []
    type_list: List[int] = []

    # 先收集所有正向边
    with open(edges_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("_label", "").upper()
            etype = EDGE_TYPE_MAP_DIRECTED.get(label)
            if etype is None:
                continue
            try:
                src_id = int(row["_outV"])
                dst_id = int(row["_inV"])
            except (KeyError, ValueError):
                continue
            src_idx = id_to_idx.get(src_id)
            dst_idx = id_to_idx.get(dst_id)
            if src_idx is None or dst_idx is None:
                continue
            src_list.append(src_idx)
            dst_list.append(dst_idx)
            type_list.append(etype)

    # 对 AST 边随机采样,只保留 30%(逻辑不变)
    filtered = [
        (s, d, t) for s, d, t in zip(src_list, dst_list, type_list)
        if t != 3 or random.random() < 0.3  # ★ AST 现在是 3,不是 2
    ]
    if filtered:
        src_list, dst_list, type_list = zip(*filtered)
        src_list, dst_list, type_list = list(src_list), list(dst_list), list(type_list)
    else:
        src_list, dst_list, type_list = [], [], []

    # ★ 新增:为有向边添加反向边(AST 除外,AST 是对称的)
    rev_src, rev_dst, rev_type = [], [], []
    for s, d, t in zip(src_list, dst_list, type_list):
        if t == 3:  # AST 不加反向
            continue
        rev_src.append(d)  # 反转方向
        rev_dst.append(s)
        rev_type.append(t + 4)  # 反向 relation = 正向 + 3

    # 测试
    # print(f"[DEBUG] 反向边数量: {len(rev_src)}")
    # print(f"[DEBUG] 反向边类型分布:")
    # print(f"  {Counter(rev_type)}")

    src_list.extend(rev_src)
    dst_list.extend(rev_dst)
    type_list.extend(rev_type)

    # print(f"[DEBUG] 最终总边数: {len(type_list)}")
    # print(f"[DEBUG] 最终类型分布:")
    # print(f"  {Counter(type_list)}")

    return src_list, dst_list, type_list


# ─────────────────────────────────────────────────────────────────────────────
#  主类
# ─────────────────────────────────────────────────────────────────────────────

class VSAPreprocessor:
    """
    将原始数据集转换为模型可用的格式。

    输入 JSON 格式（列表，每个元素一条样本）：
    {
        "func":      "void foo(char *p) { ... }",
        "vuln_type": "UAF",
        "key_lines": [3, 7, 8]          // 可选，人工标注的关键行号（0-indexed）
    }

    参数
    ----
    model_name    : HuggingFace tokenizer 名称
    max_seq_len   : token 序列最大长度
    joern_home    : Joern 安装根目录。None 时从环境变量 JOERN_HOME 读取。
                    若最终仍无法定位 joern.bat，则 _build_graph 使用规则化 fallback。
    joern_timeout : 单个函数调用 Joern 的超时秒数（默认 60）
    """

    def __init__(
            self,
            model_name: str = "./CodeBert-pretrained",
            max_seq_len: int = 512,
            joern_home: str | None = None,
            joern_timeout: int = 120,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_len = max_seq_len
        self.joern_timeout = joern_timeout
        self.sys = platform.system()

        if self.sys == "Windows":
            print("当前数据处理环境为Windows，搜索joern.bat文件...")
            # 定位 joern.bat（仅 Windows）
            home = os.environ.get("JOERN_HOME", "") or joern_home
            candidate = Path(home) / "joern.bat" if home else None
            self.joern_bin: Path | None = (
                candidate if (candidate and candidate.is_file()) else None
            )
            if self.joern_bin:
                print(f"[VSAPreprocessor] Joern 已就绪：{self.joern_bin}")
            else:
                print("[VSAPreprocessor] 未找到 joern.bat，将使用规则化 PDG fallback。")
        elif self.sys == "Linux":
            print("当前数据处理环境为Linux，搜索joern文件...")
            # 定位 joern（仅 Linux），建议配置JOERN_HOME和path，不建议手动输入目录
            home = os.environ.get("JOERN_HOME", "") or joern_home
            candidate = Path(home) / "joern" if home else None
            self.joern_bin: Path | None = (
                candidate if (candidate and candidate.is_file()) else None
            )
            if self.joern_bin:
                print(f"[VSAPreprocessor] Joern 已就绪：{self.joern_bin}")
            else:
                print("[VSAPreprocessor] 未找到 joern，将使用规则化 PDG fallback。")

    def _has_trigger(self, code: str, vuln_type: str) -> bool:
        """检查源码是否含至少一个该漏洞类型的 trigger keyword。"""
        if vuln_type == "clean" or vuln_type not in VULN_PATTERNS:
            return True
        triggers = VULN_PATTERNS[vuln_type].get("trigger_keywords", [])
        return any(t in code for t in triggers)

    # ── 对外接口 ──────────────────────────────────────────────────────────────

    def process_file(self, input_path: str, output_path: str) -> None:
        """处理整个数据集文件,保存为 .pt 列表。包含训练数据清洗。"""
        samples: List[dict] = []
        skipped_short = 0
        skipped_no_trigger = 0
        relabeled_to_clean = 0

        with open(input_path, encoding="utf-8") as f:
            raw: List[dict] = json.load(f)

        for item in tqdm(raw, desc="preprocessing"):
            source = item.get("source", "sard")

            result = self.process_one(item)
            if result is None:
                skipped_short += 1
                continue
            # 训练数据清洗:对于SARD数据集非clean样本必须有至少一个 trigger token 命中，否则认为是没有漏洞的样本，这样可能会降低这个数据集下的准确率
            # SRTS阶段构建对称样本，临时关闭SARD数据清洗
            if source == "sard":
                if result["vuln_type"] != "clean" and result["token_mask"].sum().item() == 0:
                    skipped_no_trigger += 1
                    continue
                if result["vuln_type"] != "clean" and not self._has_trigger(item["func"], result["vuln_type"]):
                    result["vuln_type"] = "clean"
                    result["label"] = torch.tensor(LABEL_MAP["clean"], dtype=torch.long)
                    relabeled_to_clean += 1

            samples.append(result)

        torch.save(samples, output_path)
        print(f"保存 {len(samples)} 条")
        print(f"  - 跳过(代码过短): {skipped_short}")
        print(f"  - 跳过(无任何标记,疑似 source 函数): {skipped_no_trigger}")
        print(f"  - 重标为 clean(无 trigger，但是有其他标记的伪漏洞样本): {relabeled_to_clean}")

    def process_one(self, item: dict) -> dict | None:
        """处理单条样本，返回 dict；失败时返回 None。"""
        vuln_type = item.get("vuln_type", "clean")
        # 显式跳过已被废弃的漏洞类型，因为不想重新更新数据集
        if vuln_type in DEPRECATED_TYPES:
            return None

        code = item["func"].strip()

        label = LABEL_MAP.get(vuln_type, 0)
        key_lines: List[int] | None = item.get("key_lines", None)

        lines = code.split("\n")
        if len(lines) < 3:
            return None

        # Step 1: tokenize
        encoding = self.tokenizer(
            code,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].squeeze(0)  # (L,)
        attention_mask = encoding["attention_mask"].squeeze(0)  # (L,)
        offset_mapping = encoding["offset_mapping"].squeeze(0)  # (L, 2)

        # Step 2: 行级弱标注掩码
        if key_lines is not None:
            line_mask = self._manual_line_mask(lines, key_lines)
        else:
            line_mask = self._auto_line_mask(lines, vuln_type)

        # Step 3: 行级掩码 → token 级掩码
        token_mask = self._align_mask_to_tokens(
            lines, line_mask, offset_mapping, attention_mask, vuln_type
        )

        # Step 4: 构建 PDG（优先 Joern，失败则 fallback）
        graph = self._build_graph(code, lines, line_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_mask": token_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "graph": graph,
            "vuln_type": vuln_type,
        }

    # ── Step 2a: 人工标注行掩码 ───────────────────────────────────────────────

    def _manual_line_mask(
            self, lines: List[str], key_lines: List[int]
    ) -> List[float]:
        mask = [0.0] * len(lines)
        for idx in key_lines:
            if 0 <= idx < len(lines):
                mask[idx] = 1.0
        return mask

    # ── Step 2b: 关键词规则生成弱标注掩码 ────────────────────────────────────

    def _auto_line_mask(self, lines: List[str], vuln_type: str) -> List[float]:
        """
        规则流程：
          1. 触发行：含 trigger_keyword 的行 → mask = 1.0
          2. 上下文行：触发行前后 3 行内，含 context_keyword → mask = 1.0
             （无 context_keywords 时窗口内全部赋弱置信度 0.5）
          3. 各漏洞类型额外补充标注（见各 elif 分支）
          4. Fallback：规则完全未命中时，前 1/3 行赋低置信度 0.3

        这是弱监督的核心，质量直接影响 L_probe 的效果。
        建议运行后人工抽样检查 10–15%。
        """
        if vuln_type not in VULN_PATTERNS or vuln_type == "clean":
            return [0.0] * len(lines)

        pattern = VULN_PATTERNS[vuln_type]
        mask = [0.0] * len(lines)
        trigger_indices: List[int] = []

        # ── 触发行 ────────────────────────────────────────────────────────────
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(kw in stripped for kw in pattern["trigger_keywords"]):
                mask[i] = 1.0
                trigger_indices.append(i)

        # ── 上下文行（触发行前后 5 行） ───────────────────────────────────────
        ctx_kws = pattern.get("context_keywords", [])
        context_window = 5
        for ti in trigger_indices:
            start = max(0, ti - context_window)
            end = min(len(lines), ti + context_window + 1)
            for j in range(start, end):
                stripped_j = lines[j].strip()
                if ctx_kws:
                    if any(kw in stripped_j for kw in ctx_kws):
                        mask[j] = 1.0
                else:
                    mask[j] = max(mask[j], 0.5)

        # ── 漏洞类型专属补充标注 ─────────────────────────────────────────────

        if vuln_type == "NullDeref":
            # 标"对指针参数解引用"的行:*ptr 或 ptr->field 形式
            import re
            deref_pat = re.compile(
                r'(?:^|[\s=,(])\*\s*\w+'  # *p 或 p->field
            )
            null_check_pat = re.compile(r'\bNULL\b|\bnullptr\b|==\s*0\b')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if deref_pat.search(stripped):
                    mask[i] = 1.0
                # NULL 比较行也是 NullDeref 的关键(漏掉检查就是漏洞)
                if null_check_pat.search(stripped):
                    mask[i] = max(mask[i], 0.7)
        elif vuln_type == "UAF":
            # UAF 的 use sink 经常远离 free,通用 ±3 窗口扫不到
            # 策略:从每个 free 行往后扫到函数末尾,标第一个出现的 sink
            sink_kws_uaf = [
                "printIntLine", "printLongLongLine", "printLine",
                "printWLine", "printHexCharLine",
                "printf", "fprintf",
            ]
            deref_patterns = ["->", "[", "*("]  # 解引用形式

            for ti in trigger_indices:
                # 从 trigger 行的下一行扫起,标记第一个 sink 或解引用
                for j in range(ti + 1, len(lines)):
                    stripped_j = lines[j].strip()
                    if any(kw in stripped_j for kw in sink_kws_uaf):
                        mask[j] = 1.0
                        break
                    # 也接受解引用模式作为 sink
                    if any(p in stripped_j for p in deref_patterns):
                        mask[j] = 1.0
                        break
        elif vuln_type == "StackBufferOverflow":
            # 栈数组声明:type name[N]; 形式
            # 严格要求方括号 + 不含 = 赋值(避免误标含初始化的全局/堆变量)
            import re
            stack_array_pat = re.compile(
                r'\b(char|int|short|long|unsigned|uint8_t|uint16_t|uint32_t|BYTE|WORD)\s+\w+\s*\[\s*[^\]]+\s*\]'
            )
            for i, line in enumerate(lines):
                if stack_array_pat.search(line) and "=" not in line.split("[")[0]:
                    mask[i] = 1.0

        elif vuln_type == "HeapBufferOverflow":
            # 标注堆分配行
            alloc_kws = [
                "malloc(", "calloc(", "realloc(",
                "kmalloc(", "kzalloc(", "vmalloc(", "new ",
            ]
            for i, line in enumerate(lines):
                if any(kw in line for kw in alloc_kws):
                    mask[i] = 1.0
            # 函数签名中含 size/len 等参数名 → 标注（可能未做边界检查）
            param_inds = pattern.get("param_indicators", [])
            for i, line in enumerate(lines[:5]):
                if any(ind in line for ind in param_inds) and "(" in line:
                    mask[i] = max(mask[i], 0.7)

        elif vuln_type == "OSCommandInjection":
            # 标注含外部输入参数的函数签名行
            param_inds = pattern.get("param_indicators", [])
            for i, line in enumerate(lines[:5]):
                if any(ind in line for ind in param_inds) and "(" in line:
                    mask[i] = 1.0
            # 标注第一个命令执行调用行（最核心的触发点，避免过度标注）
            for i, line in enumerate(lines):
                if any(kw in line for kw in pattern["trigger_keywords"]):
                    mask[i] = 1.0
                    break

        elif vuln_type == "LDAPInjection":
            # 标注含外部输入参数的函数签名行
            param_inds = pattern.get("param_indicators", [])
            for i, line in enumerate(lines[:5]):
                if any(ind in line for ind in param_inds) and "(" in line:
                    mask[i] = 1.0
            # 标注 filter/query 字符串拼接行（注入实际发生处）
            concat_kws = ["sprintf(", "snprintf(", "strcat(", "strcpy("]
            for i, line in enumerate(lines):
                if any(kw in line for kw in concat_kws):
                    mask[i] = 1.0
            # 标注第一个 LDAP 执行调用行
            for i, line in enumerate(lines):
                if any(kw in line for kw in pattern["trigger_keywords"]):
                    mask[i] = 1.0
                    break

        elif vuln_type == "IntOverflow":
            import re
            # 1. 标 atoi/strtol 等"输入转整数"调用(整数溢出的源头)
            int_input_kws = ["atoi(", "atol(", "strtol(", "strtoul(", "atoll("]
            for i, line in enumerate(lines):
                if any(kw in line for kw in int_input_kws):
                    mask[i] = 1.0

            # 2. 标含算术运算 + size_t/length 类变量名的行
            arith_pat = re.compile(r'[+\-*/]|<<|>>')
            size_var_pat = re.compile(r'\b(size|len|length|count|num|sz)\b', re.IGNORECASE)
            for i, line in enumerate(lines):
                if arith_pat.search(line) and size_var_pat.search(line):
                    mask[i] = 1.0

        # ── Fallback ──────────────────────────────────────────────────────────
        if sum(mask) == 0.0:
            n_fb = max(1, len(lines) // 3)
            for i in range(n_fb):
                mask[i] = 0.3

        return mask

    # ── Step 3: 行级掩码 → token 级掩码 ──────────────────────────────────────

    def _align_mask_to_tokens(
            self,
            lines: List[str],
            line_mask: List[float],
            offset_mapping: torch.Tensor,  # (L, 2)
            attention_mask: torch.Tensor,  # (L,)
            vuln_type: str,
    ) -> torch.Tensor:
        """
        将行级掩码精确对齐到子词 token 级别。

        新策略(关键 token 级标注):
            行级 mask 不再无差别下发到该行所有 token,而是只激活该行内
            落在 critical keyword 字符区间内的 token。这避免了符号、空格、
            变量名等"伴随 token"被错误监督。

            critical keywords 来自 VULN_PATTERNS 中该漏洞类型的:
            - trigger_keywords (主要风险操作)
            - context_keywords (上下文风险标识)
            过滤掉长度 < 2 的纯符号(它们没有跨样本的判别力)。
        """

        # ── 收集 critical keywords ─────────────────────────────────────────────
        if vuln_type in VULN_PATTERNS:
            pattern = VULN_PATTERNS[vuln_type]
            critical_kws_raw = (
                    pattern.get("trigger_keywords", [])
                    + pattern.get("context_keywords", [])
            )
            # 去掉调用括号和空白,只留函数名/操作符:"free(" → "free", "new " → "new"
            # 长度 >= 2 的 keyword 全部保留;长度 < 2 的只在白名单里保留
            SHORT_KW_WHITELIST = {"[", "->"}  # 注意 "->" 长度 2 已经在通用规则里

            critical_kws = []
            for kw in critical_kws_raw:
                cleaned = kw.rstrip("(").strip()
                # 长度 < 2 的纯符号噪声太大,过滤掉
                # (NullDeref 的 -> 等长度 == 2 的会保留)
                #if len(cleaned) >= 2 or cleaned in SHORT_KW_WHITELIST: # 这条改动有风险。[ 在普通代码里出现频率高,容易误激活。建议你先不改这条,跑一下完善 1 + 双视角池化的效果。如果 attention 仍然漏 sink,再放开 [
                if len(cleaned) >= 2:
                    critical_kws.append(cleaned)
        else:
            critical_kws = []

        # 计算每行字符起始偏移
        line_char_starts: List[int] = []
        pos = 0
        for line in lines:
            line_char_starts.append(pos)
            pos += len(line) + 1  # +1 for '\n'

        L = offset_mapping.size(0)
        token_mask = torch.zeros(L, dtype=torch.float)

        for i in range(L):
            # padding token:attention_mask == 0
            if attention_mask[i].item() == 0:
                continue

            char_start = int(offset_mapping[i, 0].item())
            char_end = int(offset_mapping[i, 1].item())

            # 特殊 token([CLS] / [SEP]):offset 恒为 (0, 0)
            if char_start == 0 and char_end == 0:
                continue

            line_idx = self._find_line(line_char_starts, char_start)
            if line_idx >= len(line_mask):
                continue

            # 该行未被弱标注命中 → token mask = 0
            if line_mask[line_idx] == 0:
                continue

            # 该行命中,但只激活落在 critical keyword 字符区间内的 token
            if not critical_kws:
                # 漏洞类型不在 VULN_PATTERNS 中(如 clean),保留行级行为兜底
                token_mask[i] = line_mask[line_idx]
                continue

            line_start_char = line_char_starts[line_idx]
            line_text = lines[line_idx]
            # token 在本行内的字符区间 [tok_s, tok_e)
            tok_s = char_start - line_start_char
            tok_e = char_end - line_start_char

            # 检查该 token 是否与本行内任意 critical keyword 的出现位置有重叠
            is_critical = False
            for kw in critical_kws:
                search_pos = 0
                while True:
                    kw_pos = line_text.find(kw, search_pos)
                    if kw_pos == -1:
                        break
                    kw_end = kw_pos + len(kw)
                    # 区间重叠判定:NOT (token 在 kw 之前 OR token 在 kw 之后)
                    if not (tok_e <= kw_pos or tok_s >= kw_end):
                        is_critical = True
                        break
                    search_pos = kw_end
                if is_critical:
                    break

            if is_critical:
                token_mask[i] = line_mask[line_idx]

        return token_mask

    @staticmethod
    def _find_line(line_starts: List[int], char_pos: int) -> int:
        """二分查找 char_pos 所在行号。"""
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= char_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # ── Step 4: 构建程序依赖图（PDG） ────────────────────────────────────────

    def _build_graph(
            self,
            code: str,
            lines: List[str],
            line_mask: List[float],
    ) -> Data:
        """
        构建 PDG。优先调用 Joern生成精确图；
        若 Joern 不可用或执行失败，自动 fallback 到规则化方法。
        """
        if self.joern_bin is not None:
            try:
                return self._build_graph_joern(code, lines, line_mask)
            except Exception as exc:
                print(f"[_build_graph] Joern 失败（{exc}），使用规则化 fallback。")
        return self._build_graph_rule(lines, line_mask)

    # ── Step 4a: Joern 路径 ───────────────────────────────────────────────────

    def _build_graph_joern(
            self,
            code: str,
            lines: List[str],
            line_mask: List[float],
    ) -> Data:
        """
        调用 Joern 构建真实 CPG/PDG。

        流程：
          1. 将函数代码写入临时 .c 文件
          2. 生成 Joern Scala 导出脚本（输出 nodes.csv / edges.csv）
          3. 通过 cmd /c joern.bat --script 执行脚本
          4. 解析 CSV，构建 PyG Data
          5. 临时目录自动清理
        """
        with tempfile.TemporaryDirectory(prefix="vsa_joern_") as tmpdir:
            tmp = Path(tmpdir)
            src_file = tmp / "func.c"
            out_dir = tmp / "out"
            script_path = tmp / "export.sc"

            out_dir.mkdir()
            src_file.write_text(code, encoding="utf-8")

            script_content = self._make_joern_script(str(src_file), str(out_dir))
            # print("=== SCRIPT ===")
            # print(script_content)
            script_path.write_text(script_content, encoding="utf-8")

            # Windows 下通过 cmd /c 启动joern.bat，Linux下直接通过命令行启动joern
            # 设置 JAVA_TOOL_OPTIONS 强制 UTF-8，避免中文路径或注释导致乱码
            env = os.environ.copy()
            env["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8 -Dstdout.encoding=UTF-8"
            joern_cmd = ''
            if self.sys == "Windows":
                joern_cmd = f'chcp 65001 >nul && "{self.joern_bin}" --script "{script_path}"'
            elif self.sys == "Linux":
                joern_cmd = f'"{self.joern_bin}" --script "{script_path}"'

            proc = subprocess.run(
                joern_cmd,
                shell=True,
                capture_output=True,
                timeout=self.joern_timeout,
                env=env,
                cwd=str(tmp),
            )

            def safe_decode(b: bytes) -> str:
                for enc in ("utf-8", "gbk", "latin-1"):
                    try:
                        return b.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return b.decode("latin-1")

            stdout = safe_decode(proc.stdout)
            stderr = safe_decode(proc.stderr)

            if proc.returncode != 0:
                raise RuntimeError(
                    f"joern 退出码 {proc.returncode}\n"
                    f"stderr: {proc.stderr[:400]}"
                )

            nodes_csv = out_dir / "nodes.csv"
            edges_csv = out_dir / "edges.csv"
            if not nodes_csv.exists() or not edges_csv.exists():
                raise FileNotFoundError(
                    "Joern 脚本未生成预期的 CSV 文件。\n"
                    f"stdout: {proc.stdout[:300]}"
                )

            # 解析节点与边
            id_to_idx, node_codes, node_lines = _parse_joern_nodes(nodes_csv)
            n = len(node_codes)
            if n == 0:
                raise ValueError("Joern 未解析到有效节点。")

            # 按行截断到 max_nodes（Joern 图可能远大于规则化图）
            max_nodes = 256
            '''
            if n > max_nodes:
                keep_ids = set(list(id_to_idx.keys())[:max_nodes])
                id_to_idx = {k: v for k, v in id_to_idx.items() if k in keep_ids}
                node_codes = node_codes[:max_nodes]
                node_lines = node_lines[:max_nodes]
                n = max_nodes
            '''
            # 按行截断修改
            if n > max_nodes:
                # 按 (lineNumber, 原始索引) 升序排,未知行号(-1)放到最后
                order = sorted(
                    range(n),
                    key=lambda i: (node_lines[i] if node_lines[i] > 0 else float("inf"), i),
                )
                keep_order = sorted(order[:max_nodes])  # 保留前 max_nodes 个,再按原索引排回去保稳定

                # 重建 node_codes / node_lines
                node_codes = [node_codes[i] for i in keep_order]
                node_lines = [node_lines[i] for i in keep_order]

                # 重建 id_to_idx:旧索引 → 新索引的映射
                old_to_new = {old: new for new, old in enumerate(keep_order)}
                id_to_idx = {
                    nid: old_to_new[old_idx]
                    for nid, old_idx in id_to_idx.items()
                    if old_idx in old_to_new
                }
                n = max_nodes

            src_list, dst_list, type_list = _parse_joern_edges(edges_csv, id_to_idx)

            # 节点特征
            node_feats = self._compute_node_features(node_codes)  # (n, 128)

            # ★ 新增:节点角色标签 ──────────────────────────────────────────
            trigger_mask = torch.zeros(n, dtype=torch.bool)
            sink_mask = torch.zeros(n, dtype=torch.bool)
            trigger_subtype = torch.zeros(n, dtype=torch.long)

            for i, code in enumerate(node_codes):
                is_trigger, is_sink = _classify_node_role(code)
                trigger_mask[i] = is_trigger
                sink_mask[i] = is_sink
                if is_trigger:
                    trigger_subtype[i] = _classify_trigger_subtype(code)
            # ───────────────────────────────────────────────────────────────
            # 临时增加检查trigger情况
            '''
            trigger_count = trigger_mask.sum().item()
            if trigger_count > 0:  # 异常多
                print(f"⚠ 该函数 trigger 节点数 {trigger_count}:")
                for i, code in enumerate(node_codes):
                    if trigger_mask[i]:
                        print(f"   [{i}] {code}")
            '''

            # key_mask：通过 Joern 报告的行号对齐
            key_mask = self._joern_key_mask(node_lines, line_mask, len(lines), n)

            if not src_list:
                src_list = [0]
                dst_list = [0]
                type_list = [1]

            # print(f"[DEBUG] Joern 路径,节点数={n},边数={len(src_list)}")

            return VSAGraphData(
                x=node_feats,
                edge_index=torch.tensor([src_list, dst_list], dtype=torch.long),
                edge_type=torch.tensor(type_list, dtype=torch.long),
                key_mask=key_mask,
                trigger_mask=trigger_mask,  # ★ 新增
                sink_mask=sink_mask,  # ★ 新增
                trigger_subtype=trigger_subtype,
                num_nodes=n,
            )

    @staticmethod
    def _make_joern_script(src_path: str, out_dir: str) -> str:
        """
        生成 Joern Scala 脚本，将 CPG 节点和边以 TSV 格式写出。
        兼容 Joern 4.0（flatgraph 后端，OverflowDB 已移除）。

        输出文件：
          <out_dir>/nodes.csv  — 列：id / _label / code / lineNumber
          <out_dir>/edges.csv  — 列：_outV / _inV / _label
        """
        src_e = src_path.replace("\\", "/")
        out_e = out_dir.replace("\\", "/")

        return f"""\
    importCode(inputPath="{src_e}", projectName="vsa_tmp")

    def esc(s: String): String = s.replace("\\t", " ").replace("\\n", " ").replace("\\r", " ")

    val nodesOut = better.files.File("{out_e}/nodes.csv")
    nodesOut.overwrite("id\\t_label\\tcode\\tlineNumber\\n")

    cpg.call.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tCALL\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.identifier.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tIDENTIFIER\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.literal.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tLITERAL\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.parameter.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tMETHOD_PARAMETER_IN\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.methodReturn.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tMETHOD_RETURN\\t${{esc(n.typeFullName)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.controlStructure.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tCONTROL_STRUCTURE\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.block.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tBLOCK\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.local.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tLOCAL\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}
    cpg.ret.foreach {{ n =>
      nodesOut.append(s"${{n.id}}\\tRETURN\\t${{esc(n.code)}}\\t${{n.lineNumber.getOrElse("")}}\\n")
    }}

    val edgesOut = better.files.File("{out_e}/edges.csv")
    edgesOut.overwrite("_outV\\t_inV\\t_label\\n")

    cpg.all.foreach {{ node =>
      node._astOut.foreach {{ dst =>
        edgesOut.append(s"${{node.id}}\\t${{dst.id}}\\tAST\\n")
      }}
      node._cfgOut.foreach {{ dst =>
        edgesOut.append(s"${{node.id}}\\t${{dst.id}}\\tCFG\\n")
      }}
      node._cdgOut.foreach {{ dst =>
        edgesOut.append(s"${{node.id}}\\t${{dst.id}}\\tCDG\\n")
      }}
      node._reachingDefOut.foreach {{ dst =>
        edgesOut.append(s"${{node.id}}\\t${{dst.id}}\\tREACHING_DEF\\n")
      }}
    }}

    println("joern_export_done")
    """

    @staticmethod
    def _joern_key_mask(
            node_lines: List[int],
            line_mask: List[float],
            total_lines: int,
            n: int,
    ) -> torch.Tensor:
        """
        通过 Joern 节点的行号字段，将行级 line_mask 映射到节点级 key_mask。

        node_lines 中行号为 1-indexed（Joern 惯例），-1 表示未知。
        """
        key_mask = torch.zeros(n, dtype=torch.float)
        for idx, lineno in enumerate(node_lines):
            if lineno < 1:
                continue
            li = lineno - 1  # 转为 0-indexed
            if li < len(line_mask):
                key_mask[idx] = line_mask[li]
        return key_mask

    # ── Step 4b: 规则化 fallback ──────────────────────────────────────────────

    def _build_graph_rule(
            self,
            lines: List[str],
            line_mask: List[float],
    ) -> Data:
        """
        规则化 PDG 构建（Joern 不可用时的 fallback）。
          - 每行作为一个节点
          - CDG：顺序相邻行之间双向边（edge_type=1）
          - DDG：变量定义行 → 变量使用行（edge_type=0）

        节点特征：128 维（64 维关键词 one-hot + 1 维 mask 值 + 63 维位置编码）。
        """
        max_nodes = 64
        lines = lines[:max_nodes]
        line_mask = line_mask[:max_nodes]
        n = len(lines)

        node_feats = self._compute_node_features(lines, line_mask)

        src_edges: List[int] = []
        dst_edges: List[int] = []
        edge_types: List[int] = []

        # CFG forward:源码顺序(语义上是 i → i+1)
        for i in range(n - 1):
            # 正向 CFG
            src_edges.append(i)
            dst_edges.append(i + 1)
            edge_types.append(2)  # CFG forward
            # 反向 CFG
            src_edges.append(i + 1)
            dst_edges.append(i)
            edge_types.append(6)  # CFG backward

        # DDG:变量定义 → 使用(已有逻辑,但 relation 编号要确认)
        var_def_lines = self._find_var_definitions(lines)
        for var, def_indices in var_def_lines.items():
            pat = re.compile(r'\b' + re.escape(var) + r'\b')
            for def_idx in def_indices:
                for use_idx, use_line in enumerate(lines):
                    if use_idx != def_idx and pat.search(use_line):
                        # DDG forward:定义 → 使用
                        src_edges.append(def_idx)
                        dst_edges.append(use_idx)
                        edge_types.append(0)  # DDG forward
                        # DDG backward:使用 → 定义
                        src_edges.append(use_idx)
                        dst_edges.append(def_idx)
                        edge_types.append(4)  # DDG backward

        # ★ 新增:节点角色标签 ──────────────────────────────────────────
        trigger_mask = torch.zeros(n, dtype=torch.bool)
        sink_mask = torch.zeros(n, dtype=torch.bool)
        trigger_subtype = torch.zeros(n, dtype=torch.long)

        for i, code in enumerate(line_mask):
            is_trigger, is_sink = _classify_node_role(code)
            trigger_mask[i] = is_trigger
            sink_mask[i] = is_sink
            if is_trigger:
                trigger_subtype[i] = _classify_trigger_subtype(code)
        # ───────────────────────────────────────────────────────────────

        key_node_mask = torch.tensor(line_mask, dtype=torch.float)

        if not src_edges:
            src_edges = [0]
            dst_edges = [0]
            edge_types = [1]

        # print(f"[DEBUG] Rule 路径,节点数={n},边数={len(src_edges)}")

        return VSAGraphData(
            x=node_feats,
            edge_index=torch.tensor([src_edges, dst_edges], dtype=torch.long),
            edge_type=torch.tensor(edge_types, dtype=torch.long),
            key_mask=key_node_mask,
            trigger_mask=trigger_mask,  # ★ 新增
            sink_mask=sink_mask,  # ★ 新增
            trigger_subtype=trigger_subtype,
            num_nodes=n,
        )

    # ── 节点特征计算 ──────────────────────────────────────────────────────────

    def _compute_node_features(
            self,
            codes: List[str],
            line_mask: List[float] | None = None,
    ) -> torch.Tensor:
        """
        计算 128 维节点特征：
          [0  : 64) — 关键词 one-hot（_NODE_KEYWORDS 前 64 项）
          [64]      — line_mask 值（若提供；Joern 路径通过 key_mask 单独传入，此维置 0）
          [65 :128) — 正弦位置编码（向量化实现，替代原版逐 token 循环）
        """
        n = len(codes)
        feat_dim = 128
        kw_dim = _KW_DIM
        feats = torch.zeros(n, feat_dim)

        # ── 关键词 one-hot ────────────────────────────────────────────────────
        keywords = _NODE_KEYWORDS[:kw_dim]
        for i, code in enumerate(codes):
            lower = code.lower()
            for j, kw in enumerate(keywords):
                if kw in lower:
                    feats[i, j] = 1.0
            if line_mask is not None and i < len(line_mask):
                feats[i, kw_dim] = line_mask[i]

        # ── 向量化正弦位置编码 ────────────────────────────────────────────────
        pe_dim = 63  # 后 63 维（第 64 维留给 line_mask）
        pos = torch.arange(n, dtype=torch.float).unsqueeze(1)  # (n, 1)
        dim_idx = torch.arange(pe_dim, dtype=torch.float)  # (63,)
        denoms = 10000.0 ** (dim_idx / 62.0)  # (63,)
        angles = pos / denoms  # (n, 63)
        pe = torch.zeros(n, pe_dim)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        feats[:, kw_dim + 1: kw_dim + 1 + pe_dim] = pe

        return feats  # (n, 128)

    # ── 辅助：变量定义识别（DDG 构建） ───────────────────────────────────────

    def _find_var_definitions(self, lines: List[str]) -> Dict[str, List[int]]:
        """
        启发式识别 C/C++ 变量赋值行。
        匹配 `type var =` 或 `var = expr`（不含 ==）。
        过滤单字母变量（噪声多）。
        """
        var_defs: Dict[str, List[int]] = {}
        def_pattern = re.compile(
            r'(?:int|char|void|unsigned|size_t|uint\w*|struct\s+\w+|\w+_t)'
            r'\s*\*?\s*(\w+)\s*='
            r'|^\s*(\w+)\s*=[^=]',
            re.MULTILINE,
        )
        for i, line in enumerate(lines):
            for m in def_pattern.finditer(line):
                var = m.group(1) or m.group(2)
                if var and len(var) > 1:
                    var_defs.setdefault(var, []).append(i)
        return var_defs
