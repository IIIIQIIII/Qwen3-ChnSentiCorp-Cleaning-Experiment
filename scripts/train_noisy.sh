#!/bin/bash

# =================================================================
# 训练脚本: 在原始带噪的 ChnSentiCorp 数据集上微调
#
# 使用说明:
# 1. 确保已安装 LLaMA-Factory 及其依赖。
# 2. 确保基座模型 'Qwen/Qwen3-4B-Instruct-2507' 可访问。
# 3. 在项目根目录的 'scripts/' 文件夹下执行此脚本:
#    bash train_noisy.sh
# =================================================================

# 定义基座模型，可以直接从 ModelScope Hub 下载
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

# 定义数据和输出目录（使用相对路径）
DATA_DIR="../data"
OUTPUT_DIR="../saves/Qwen3-4B/freeze/sft_noisy"

# 运行 LLaMA-Factory 训练命令
llamafactory-cli train \
    --model_name_or_path ${BASE_MODEL} \
    --trust_remote_code \
    --stage sft \
    --do_train \
    --finetuning_type freeze \
    --freeze_trainable_layers 6 \
    --freeze_extra_modules embed_tokens,norm \
    --dataset sentiment_noisy \
    --dataset_dir ${DATA_DIR} \
    --template qwen3 \
    --cutoff_len 720 \
    --max_samples 10000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --dataloader_num_workers 4 \
    --output_dir ${OUTPUT_DIR} \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 50 \
    --eval_strategy steps \
    --load_best_model_at_end \
    --plot_loss \
    --overwrite_output_dir \
    --save_only_model false \
    --greater_is_better true \
    --compute_accuracy true \
    --val_size 0.2 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --fp16 \
    --ddp_timeout 180000000

echo "训练完成，模型已保存至 ${OUTPUT_DIR}"