# run_eval.py
import json

import torch
import sys
from config import VSAConfig
from model.vsa_model import VSAModel
from data.dataset import build_dataloader
from evaluate import (
    evaluate_classification,
    evaluate_attention_alignment,
    cross_project_eval,
    predict_one,
    predict_one_pt,
)
from data.preprocess import VSAPreprocessor
from data import preprocess
sys.modules['preprocess'] = preprocess


def eval_single_sample(model, test_code, device):
    # 测试 4：单样本推理
    preprocessor = VSAPreprocessor(model_name='./data/CodeBert-pretrained')
    result = predict_one(model, test_code, preprocessor, device)
    print("\n=== 单样本推理 ===")
    print(f"预测漏洞类型: {result['predicted_vuln_type']}")
    print(f"置信度:       {result['confidence']:.4f}")
    print("各类概率:")
    for vuln_type, prob in result["all_probs"].items():
        bar = "█" * int(prob * 20)
        print(f"  {vuln_type:12s}  {prob:.4f}  {bar}")
    print("\nAttention 最高的 10 个 token（模型认为最关键的位置）:")
    for token, weight in result["top_attention_tokens"]:
        print(f"  {token:15s}  {weight:.4f}")


def eval_gener_perfrom(model, device):
    # 测试 3：跨项目泛化（针对多个未参与训练项目的数据）
    print("\n=== 跨项目泛化 ===")
    cross_loaders = {
        "sard": build_dataloader("./data/processed/SARD/SARD_AUG_NO_KL_test_v6.pt", 64, False),
        "cleanvul": build_dataloader("./data/processed/CLEANVUL/CLEANVUL_NO_KEYLINE_FIXED_v6.pt", 64, False),
    }
    # "titanvul": build_dataloader("./data/processed/TITANVUL/TITANVUL_NO_KEYLINE_FIXED_v6.pt", 64, False),
    # "cleanvul": build_dataloader("./data/processed/CLEANVUL/CLEANVUL_NO_KEYLINE_FIXED_v6.pt", 64, False),
    # "primevul": build_dataloader("./data/processed/PRIMEVUL/PRIMEVUL_NO_KEYLINE_FIXED_v6.pt", 64, False),
    # "diversevul": build_dataloader("./data/processed/DIVERSEVUL/DIVERSEVUL_NO_KEYLINE_FIXED_v6.pt", 64, False),
    # "sard": build_dataloader("./data/processed/SARD/SARD_AUG_NO_KL_test_v6.pt", 64, False),
    cross_project_eval(model, cross_loaders, device)


def eval_interpretablility(model, test_loader, device, top_k=5):
    # 测试 2：Attention 对齐率（可解释性）
    align_metrics = evaluate_attention_alignment(model, test_loader, device, top_k)
    print("\n=== 可解释性（Attention 对齐）===")
    print(f"Precision@5: {align_metrics['precision@5']:.4f}")
    print(f"Recall@5:    {align_metrics['recall@5']:.4f}")


def eval_basic_perform(model, test_loader, device):
    # 测试 1：基础性能测试（F1 / Accuracy / AUC）
    metrics = evaluate_classification(model, test_loader, device)

    print("\n=== 基础性能 ===")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"AUC:         {metrics['auc']:.4f}")
    print("\n每类指标：")
    for cls_name, cls_metrics in metrics["per_class"].items():
        print(f"  {cls_name:12s}  P={cls_metrics['precision']:.3f}  "
              f"R={cls_metrics['recall']:.3f}  F1={cls_metrics['f1-score']:.3f}")


def eval_srts_pt(model, origin_path, reversal_path, device, vul_type):
    if not origin_path:
        raise ValueError("orgin_path 不能为空")
    if not reversal_path:
        raise ValueError("reversal_path 不能为空")

    origin_loader = torch.load(origin_path)
    # print(f'Origin size is {len(origin_loader)}')
    reversal_loader = torch.load(reversal_path)
    # print(f'Reversal size is {len(reversal_loader)}')

    origin_probs = []
    origin_clean = []
    origin_succ = 0
    reversal_probs = []
    reversal_clean = []
    reversal_succ = 0

    for i, preprocessed in enumerate(origin_loader):
        # print(f'Sample Index {i}')
        result = predict_one_pt(model, preprocessed, device)
        if result is None or 'all_probs' not in result:
            continue
        for vuln_type, prob in result["all_probs"].items():
            if vuln_type == vul_type:
                # print(f'原始{vuln_type}预测概率: {prob:.4f}')
                origin_probs.append(prob)
                if result['predicted_vuln_type'] == vul_type:
                    origin_succ += 1
                continue
            elif vuln_type == 'clean':
                #  print(f'原始clean预测概率: {prob:.4f}')
                origin_clean.append(prob)
                continue
            else:
                continue

    for i, preprocessed in enumerate(reversal_loader):
        # print(f'Sample Index {i}')
        result = predict_one_pt(model, preprocessed, device)
        if result is None or 'all_probs' not in result:
            continue

        for vuln_type, prob in result["all_probs"].items():
            if vuln_type == vul_type:
                # print(f'语义翻转{vuln_type}预测概率: {prob:.4f}')
                reversal_probs.append(prob)
                if result['predicted_vuln_type'] == 'clean':
                    reversal_succ += 1
                continue
            elif vuln_type == 'clean':
                # print(f'语义翻转clean预测概率: {prob:.4f}')
                reversal_clean.append(prob)
                continue
            else:
                continue

    aver_origin_prob = sum(origin_probs) / len(origin_probs)
    aver_origin_clean = sum(origin_clean) / len(origin_clean)
    # 原始成功率为判断为正确漏洞的比例
    origin_succ_rate = origin_succ / len(origin_probs)

    aver_reversal_prob = sum(reversal_probs) / len(reversal_probs)
    aver_reversal_clean = sum(reversal_clean) / len(reversal_clean)
    # 翻转成功率为判断为非漏洞的比例
    reveral_succ_rate = reversal_succ / len(origin_probs)

    # 分布统计，翻转后概率下降>0.3、0.2< <0.3、0.1< <0.2、0< <0.1、<0，5个范围
    r_1 = []  # >0.3
    r_2 = []  # 0.2< <0.3
    r_3 = []  # 0.1< <0.2
    r_4 = []  # 0< <0.1
    r_5 = []  # <0

    if len(origin_probs) != len(reversal_probs):
        print('语义翻转前后样本数量不匹配，直方图统计失败...')
    else:
        for i, ori_prob in enumerate(origin_probs):
            rev_prob = reversal_probs[i]
            diff_prob = ori_prob - rev_prob
            if diff_prob > 0.3:
                r_1.append(i)
            elif diff_prob > 0.2:
                r_2.append(i)
            elif diff_prob > 0.1:
                r_3.append(i)
            elif diff_prob > 0:
                r_4.append(i)
            else:
                r_5.append(i)

    print(
        f'当前数据集SRTS测试结果（{vul_type}）: 初始漏洞概率 {aver_origin_prob:.4f}, 语义翻转漏洞概率 {aver_reversal_prob:.4f}')
    print(f'翻转后平均预测下降概率: {aver_origin_prob - aver_reversal_prob:.4f}')
    print(
        f'当前数据集SRTS测试结果（{vul_type}）: 初始clean概率 {aver_origin_clean:.4f}, 语义翻转clean概率 {aver_reversal_clean:.4f}')
    print(f'原始漏洞准确率为: {origin_succ_rate}, 翻转安全样本准确率为: {reveral_succ_rate}')

    print('语义翻转下降直方图统计：')
    print(f'翻转后{vul_type}预测概率下降>0.3数量为{len(r_1)}，样本Index：{r_1}')
    print(f'翻转后{vul_type}预测概率下降0.2~0.3数量为{len(r_2)}，样本Index：{r_2}')
    print(f'翻转后{vul_type}预测概率下降0.1~0.2数量为{len(r_3)}，样本Index：{r_3}')
    print(f'翻转后{vul_type}预测概率下降0~0.1数量为{len(r_4)}，样本Index：{r_4}')
    print(f'翻转后{vul_type}预测概率下降<0数量为{len(r_5)}，样本Index：{r_5}')


def eval_srts(model, orgin_path, reversal_path, device, vul_type):
    if not orgin_path:
        raise ValueError("orgin_path 不能为空")
    if not reversal_path:
        raise ValueError("reversal_path 不能为空")

    with open(orgin_path, "r", encoding="utf-8") as f:
        origin_samples = json.load(f)
    with open(reversal_path, "r", encoding="utf-8") as f:
        reversal_samples = json.load(f)

    preprocessor = VSAPreprocessor(model_name='./data/CodeBert-pretrained')
    origin_probs = []
    origin_clean = []
    origin_succ = 0
    reversal_probs = []
    reversal_clean = []
    reversal_succ = 0

    index = 0
    for i, sample in enumerate(origin_samples):
        type = sample.get('vuln_type')
        if type != vul_type:
            print(f'样本类型与SRTS测试类型不符，跳过...')
            continue
        print(f'Sample Index {index}')
        index += 1
        func = sample.get('func')

        result = predict_one(model, func, preprocessor, device)
        if result is None or 'all_probs' not in result:
            continue
        for vuln_type, prob in result["all_probs"].items():
            if vuln_type == vul_type:
                print(f'原始{vuln_type}预测概率: {prob:.4f}')
                origin_probs.append(prob)
                if result['predicted_vuln_type'] == vul_type:
                    origin_succ += 1
                continue
            elif vuln_type == 'clean':
                print(f'原始clean预测概率: {prob:.4f}')
                origin_clean.append(prob)
                continue
            else:
                continue
    index = 0
    for i, sample in enumerate(reversal_samples):
        type = sample.get('vuln_type')
        if type != vul_type:
            print(f'样本类型与SRTS测试类型不符，跳过...')
            continue
        print(f'Sample Index {index}')
        index += 1
        func = sample.get('func')

        result = predict_one(model, func, preprocessor, device)
        
        if result is None or 'all_probs' not in result:
            continue
        for vuln_type, prob in result["all_probs"].items():
            if vuln_type == vul_type:
                print(f'语义翻转{vuln_type}预测概率: {prob:.4f}')
                reversal_probs.append(prob)
                if result['predicted_vuln_type'] == 'clean':
                    reversal_succ += 1
                continue
            elif vuln_type == 'clean':
                print(f'语义翻转clean预测概率: {prob:.4f}')
                reversal_clean.append(prob)
                continue
            else:
                continue

    aver_origin_prob = sum(origin_probs) / len(origin_probs)
    aver_origin_clean = sum(origin_clean) / len(origin_clean)
    # 原始成功率为判断为正确漏洞的比例
    origin_succ_rate = origin_succ / len(origin_probs)

    aver_reversal_prob = sum(reversal_probs) / len(reversal_probs)
    aver_reversal_clean = sum(reversal_clean) / len(reversal_clean)
    # 翻转成功率为判断为非漏洞的比例
    reveral_succ_rate = reversal_succ / len(origin_probs)
    print(f'当前数据集SRTS测试结果（{vul_type}）: 初始漏洞概率 {aver_origin_prob:.4f}, 语义翻转漏洞概率 {aver_reversal_prob:.4f}')
    print(f'翻转后平均预测下降概率: {aver_origin_prob - aver_reversal_prob:.4f}')
    print(f'当前数据集SRTS测试结果（{vul_type}）: 初始clean概率 {aver_origin_clean:.4f}, 语义翻转clean概率 {aver_reversal_clean:.4f}')
    print(f'原始漏洞准确率为: {origin_succ_rate}, 翻转安全样本准确率为: {reveral_succ_rate}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 加载训练好的模型 ----
    print(f'加载训练模型：DIVERSE_TITAN_PRIME_V6')
    checkpoint = torch.load("checkpoints/DIVERSE_TITAN_PRIME_V6.pt", map_location=device)
    config = checkpoint["config"]
    model = VSAModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"加载 checkpoint（来自 epoch {checkpoint['epoch']}）")
    # ---- 加载测试集 ----
    test_loader = build_dataloader("./data/processed/MIXED/TITAN_PRIME_DIVERSEVUL_val_v6.pt", batch_size=64, shuffle=False)

    # 基础性能测试（F1 / Accuracy / AUC）
    eval_basic_perform(model, test_loader, device)

    # Attention 对齐率（可解释性）
    eval_interpretablility(model, test_loader, device)

    # 跨项目泛化
    eval_gener_perfrom(model, device)

    # 单样本测试
    # get_buffer(10);
    # (int *)malloc(100*sizeof(int));
    # printIntLine(buf[0]);
    test_code1 = """
    void vuln_func(char *ptr) {
        int *buf = (int *)malloc(100*sizeof(int));
        if (!buf) return;
        free(buf);
        printIntLine(buf[0]);
    }
    """

    test_code2 = """
    ifndef buf66

    void b9zjc() {

        int64_t * data;
        data = NULL;
        if(GLOBAL_CONST_TRUE)
        {
            data = (int64_t *)malloc(100*sizeof(int64_t));
            if (data == NULL) {exit(-1);}
            {
                size_t i;
                for(i = 0; i < 100; i++)
                {
                    data[i] = 5LL;
                }
            }
            free(data);
        }
        if(GLOBAL_CONST_TRUE)
        {
            printLongLongLine(data[0]);
        }
    }
    """

    eval_single_sample(model, test_code1, device)
    eval_single_sample(model, test_code2, device)

    # 语义翻转测试

    print('语义反转测试：SARD数据集------------')
    eval_srts_pt(model, './data/processed/SRTS/UAF/SARD/UAF_ORIGIN.pt',
                 './data/processed/SRTS/UAF/SARD/UAF_REVERSAL.pt', device, 'UAF')

    print('语义反转测试：PRIMEVUL数据集------------')
    eval_srts_pt(model, './data/processed/SRTS/UAF/PRIMEVUL/UAF_ORIGIN.pt',
                 './data/processed/SRTS/UAF/PRIMEVUL/UAF_REVERSAL.pt', device, 'UAF')

    print('语义反转测试：TITAN数据集------------')
    eval_srts_pt(model, './data/processed/SRTS/UAF/TITANVUL/UAF_ORIGIN.pt',
                 './data/processed/SRTS/UAF/TITANVUL/UAF_REVERSAL.pt', device, 'UAF')

    '''
    print('语义反转测试：SARD数据集------------')
    origin_path = './data/SRTS/UAF/SARD/UAF_ORIGIN_V2.json'
    reversal_path = './data/SRTS/UAF/SARD/UAF_REVERSAL_V2.json'
    eval_srts(model, origin_path, reversal_path, device, 'UAF')
    
    print('语义反转测试：PRIMEVUL数据集------------')
    origin_path = './data/SRTS/UAF/PRIMEVUL/UAF_ORIGIN.json'
    reversal_path = './data/SRTS/UAF/PRIMEVUL/UAF_REVERSAL.json'
    eval_srts(model, origin_path, reversal_path, device, 'UAF')
    
    print('语义反转测试：TITAN数据集------------')
    origin_path = './data/SRTS/UAF/TITANVUL/UAF_ORIGIN.json'
    reversal_path = './data/SRTS/UAF/TITANVUL/UAF_REVERSAL.json'
    eval_srts(model, origin_path, reversal_path, device, 'UAF')
    '''


if __name__ == "__main__":
    main()
