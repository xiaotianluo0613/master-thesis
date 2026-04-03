#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p core
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -J svea_basic
#SBATCH --output=my_experiments/log_%j.out   # 确保 my_experiments 文件夹存在！
#SBATCH --error=my_experiments/err_%j.err

# ================= 配置区 =================
# 1. 基础路径
BASE_DIR="/proj/uppmax2025-2-505/xilu1878"

# 2. 代码位置
CODE_DIR="$BASE_DIR/svea_embedding"

# 3. 结果位置 (修改：我帮你把路径建得更深一点，分类更清晰)
RESULT_DIR="$BASE_DIR/my_experiments/basic_evaluation/run_${SLURM_JOB_ID}"

# 4. 环境位置
ENV_PATH="$BASE_DIR/envs/svea"
# ===========================================

# 1. 准备环境
module load conda
source activate $ENV_PATH

# 创建结果文件夹
mkdir -p $RESULT_DIR
echo "=== Results will be stored in: $RESULT_DIR ==="

# 2. 进入代码目录
cd $CODE_DIR

echo "=== Running Evaluation ==="

# 3. 运行评估
# 【重要修正】这里改成了 example_gold_standard.jsonl
python evaluation.py \
    --test_data_file example_pos.jsonl \
    --goldfile example_gold_standard.jsonl \
    --k_documents 100 \
    --methods \
        bm25 \
        BAAI/bge-m3 \
        KBLab/sentence-bert-swedish-cased \
        castorini/mdpr-tied-pft-msmarco \
        gemasphi/mcontriever \
        intfloat/multilingual-e5-small

# 4. 收尾工作
echo "=== Moving results ==="
if [ -f "scores.txt" ]; then
    mv scores.txt "$RESULT_DIR/scores.txt"
    echo "Success! Scores moved to $RESULT_DIR/scores.txt"
else
    echo "WARNING: scores.txt was not found!"
fi

# 尝试把日志文件复制进去（因为还在写入，用 cp 比 mv 安全）
cp "$BASE_DIR/my_experiments/log_${SLURM_JOB_ID}.out" "$RESULT_DIR/run.log" 2>/dev/null
cp "$BASE_DIR/my_experiments/err_${SLURM_JOB_ID}.err" "$RESULT_DIR/run.err" 2>/dev/null

echo "=== All Done! ==="