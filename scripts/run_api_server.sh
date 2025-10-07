#!/bin/bash

# =================================================================
# API 服务启动脚本
#
# 默认加载 ModelScope 上性能最佳的模型（基于清洗后数据训练）。
# 你也可以取消注释并修改路径，以加载本地训练好的模型。
#
# 使用说明:
# 1. 确保已安装 LLaMA-Factory 及其 vLLM 依赖。
# 2. 在项目根目录的 'scripts/' 文件夹下执行此脚本:
#    bash run_api_server.sh
# =================================================================

# 选项 1: 加载 ModelScope Hub 上的模型 (默认)
MODEL_PATH="mc36473/Qwen3-4B-ChnSentiCorp-Sentiment"

# 选项 2: 加载本地训练好的模型 (如果需要，请取消下面的注释)
# MODEL_PATH="../saves/Qwen3-4B/freeze/sft_clean"

# 设置 API 端口
API_PORT=8000

echo "正在启动 API 服务..."
echo "模型路径: ${MODEL_PATH}"
echo "API 将在 http://localhost:${API_PORT} 上可用"

# 启动服务
API_PORT=${API_PORT} llamafactory-cli api \
    --model_name_or_path ${MODEL_PATH} \
    --template qwen3 \
    --infer_backend vllm \
    --trust_remote_code