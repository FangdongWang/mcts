#!/bin/bash
source /home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/venv_mcts/bin/activate
cd /home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2

# 定义需要遍历的文件路径列表
# data_lists=(
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/refcocog.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/docvqa.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/chartqa_single.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/textvqa.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/RefCOCO+.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/info_vqa_single.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/refcoco.jsonl"
# )

# data_lists=(
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/docvqa.jsonl" 1
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/chartqa_single.jsonl" 1
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/textvqa.jsonl" 本地
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/RefCOCO+.jsonl" 本地
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/info_vqa_single.jsonl" 本地
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/refcoco.jsonl" 本地
# )
# 换成了seed-vl
# data_lists=(
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/info_vqa_single.jsonl" 
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/refcoco.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/RefCOCO+.jsonl"
#     "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0915/chartqa_single.jsonl" 
# )
data_lists=(
    # "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/refcocog.jsonl"
    # "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/docvqa.jsonl"
    # "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/chartqa_single.jsonl"
    # "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/textvqa.jsonl"
    "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/intent.jsonl"
    # "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/info_vqa_single.jsonl"
    # "/home/fangdong.wang/mlm-evaluator/tools/mcts_cot_v2/data/sample_data_0926/refcoco.jsonl"
)




# 获取总文件数
total=${#data_lists[@]}

echo "共需运行 $total 个文件..."
echo "-------------------------"

# 遍历所有文件路径
for index in "${!data_lists[@]}"; do
    # 当前文件路径
    file_path="${data_lists[$index]}"
    # 计算当前是第几个（索引从0开始，+1转为人类可读的序号）
    current=$((index + 1))
    
    echo "第 $current/$total 个文件：$file_path"
    echo "开始运行：python test.py --qaf \"$file_path\""
    
    # 执行test.py，传入当前文件路径作为-qaf参数
    python test.py --qaf "$file_path"
    
    # 检查上一条命令的执行结果（$?为0表示成功，非0表示失败）
    if [ $? -eq 0 ]; then
        echo "✅ 运行成功"
    else
        echo "❌ 运行失败"
    fi
    
    echo "-------------------------"
done

echo "所有文件运行完毕！"
