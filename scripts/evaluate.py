#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChnSentiCorp 情感分析评测脚本（逐条请求）。

该脚本逐一读取测试集中的文本，调用本地运行的 LLM API 服务进行情感预测，
并将结果与真实标签进行比较，最后输出详细的分类评估报告。

使用方法:
1. 首先，确保已通过 run_api_server.sh 启动了推理服务。
2. 在项目根目录的 'scripts/' 文件夹下执行此脚本:
   cd scripts/
   python evaluate.py
   # 或者从项目根目录执行:
   # python scripts/evaluate.py
"""
import json
import logging
import time
from typing import Optional

import pandas as pd
import requests
from sklearn.metrics import (accuracy_score, classification_report)
from tqdm import tqdm

# --- 配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("evaluate.log")],
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "qwen3"
MAX_RETRY = 3
TIMEOUT = 60
SAVE_INTERVAL = 50  # 每处理 N 条数据就保存一次结果

# ------------------------------------------------------------------
# 工具函数: 构造 prompt / 解析返回
# ------------------------------------------------------------------
def build_prompt(text: str) -> str:
    """构造发送给模型的标准 prompt。"""
    return f"""请对以下中文文本进行情感分析，判断其情感倾向。

任务说明：
- 分析文本表达的整体情感态度
- 判断是正面(1)还是负面(0)

文本内容：
```
{text}
```

输出格式：
```json
{{
  "sentiment": 0 or 1
}}
```

请直接输出JSON结果，不要包含任何解释或其他文本。"""


def parse_sentiment(raw: str) -> Optional[int]:
    """
    从模型返回的字符串中解析出情感标签 (0 或 1)。
    
    支持多种格式，如 markdown 的 json 代码块或纯 json。
    如果解析失败或值无效，则返回 None。
    """
    try:
        # 移除 markdown 代码块标记
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        
        doc = json.loads(raw.strip())
        val = int(doc["sentiment"])
        
        return val if val in (0, 1) else None
    except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
        logger.warning(f"解析JSON失败: {e}. 原始文本: '{raw[:100]}...'")
        # 尝试最后的兜底策略
        if '"sentiment": 1' in raw or '"sentiment": "1"' in raw:
            return 1
        if '"sentiment": 0' in raw or '"sentiment": "0"' in raw:
            return 0
        return None

# ------------------------------------------------------------------
# 单条 API 调用
# ------------------------------------------------------------------
def call_api_single(text: str) -> Optional[int]:
    """
    向 API 发送单条文本进行情感分析，并包含重试逻辑。
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": build_prompt(text)}],
        "temperature": 0.0,
        "stream": False,
    }
    
    for attempt in range(MAX_RETRY):
        try:
            resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()  # 如果状态码不是 2xx，则抛出异常
            choice = resp.json()["choices"][0]
            return parse_sentiment(choice["message"]["content"])
        except requests.RequestException as e:
            logger.warning(f"请求失败 (尝试 {attempt+1}/{MAX_RETRY}): {e}")
            time.sleep(2 ** attempt)  # 指数退避
        except (KeyError, IndexError) as e:
            logger.error(f"API返回格式错误 (尝试 {attempt+1}/{MAX_RETRY}): {e}. 响应: {resp.text}")
            return None # 格式错误通常无法通过重试解决
            
    logger.error(f"文本 '{text[:50]}...' 在 {MAX_RETRY} 次尝试后仍无法获取预测结果。")
    return None

# ------------------------------------------------------------------
# 评测主流程
# ------------------------------------------------------------------
def evaluate_csv(csv_path: str, output_csv: str):
    """
    读取 CSV 文件，逐条进行推理，并计算最终评估指标。
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"测试文件未找到: {csv_path}. 请确保文件路径正确。")
        return

    assert {"text_a", "label"}.issubset(df.columns), "CSV文件必须包含 'text_a' 和 'label' 列。"
    
    df["predicted_label"] = None
    df["success"] = False

    logger.info(f"开始对 {len(df)} 条数据进行评测...")

    # 使用 tqdm 创建进度条
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Inferring"):
        text = str(row["text_a"])
        prediction = call_api_single(text)
        
        df.loc[index, "predicted_label"] = prediction
        df.loc[index, "success"] = prediction is not None

        # 定期保存，防止意外中断
        if (index + 1) % SAVE_INTERVAL == 0:
            df.to_csv(output_csv, index=False, encoding="utf-8")

    # 最终保存
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"推理完成，详细结果已写入 {output_csv}")

    # 过滤掉预测失败的样本
    successful_df = df.dropna(subset=['predicted_label'])
    if successful_df.empty:
        logger.error("所有样本均未能成功预测，无法计算指标。")
        return
        
    y_true = successful_df["label"].astype(int)
    y_pred = successful_df["predicted_label"].astype(int)
    
    # 打印评估报告
    print("\n" + "=" * 60)
    print("ChnSentiCorp 情感分析评测结果")
    print("=" * 60)
    print(f"成功样本: {len(successful_df)}/{len(df)} ({len(successful_df)/len(df)*100:.2f}%)")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("-" * 60)
    print("详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=["负面 (0)", "正面 (1)"], digits=4))
    print("=" * 60)

# ------------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChnSentiCorp 情感分析评测脚本")
    parser.add_argument(
        "--csv", 
        default="../data/test/ChnSentiCorp_test.csv", 
        help="输入测试集的 CSV 文件路径"
    )
    parser.add_argument(
        "--out", 
        default="evaluation_results.csv", 
        help="保存带有预测结果的 CSV 文件名"
    )
    args = parser.parse_args()
    
    evaluate_csv(args.csv, args.out)