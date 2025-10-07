# 通过清洗标签噪声提升 ChnSentiCorp 情感分析效果

[![ModelScope](https://img.shields.io/badge/ModelScope-Qwen3--4B--ChnSentiCorp--Sentiment-blue)](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-Sentiment/summary)
[![ModelScope](https://img.shields.io/badge/ModelScope-Qwen3--4B--ChnSentiCorp--ACC97-gray)](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-ACC97/summary)

[ [English](README.md) | 中文 ]

---

本项目旨在记录一个实验：通过清洗中文情感分析基准数据集 **ChnSentiCorp** 中存在的标签噪声，来提升模型在该任务上的性能。我们使用 **Qwen3-4B** 作为基座模型，并通过 LLaMA-Factory 在原始（带噪）和清洗后的数据集上分别进行了微调。

实验结果表明，在清洗后的数据集上训练，可以显著提升模型的准确率和 F1 分数。

## 核心发现

在清洗后的数据集上训练，模型在测试集上的准确率获得了 **1.67%** 的绝对提升。

| 训练数据集 | 准确率 | 宏平均-精确率 | 宏平均-召回率 | 宏平均-F1 |
| :------- | :----: | :---------: | :---------: | :-------: |
| 原始带噪数据 | 97.21% |   97.14%    |   97.31%    |  97.20%   |
| **清洗后数据** | **98.88%** | **98.88%** | **98.88%** | **98.88%** |


## 模型下载

我们微调了两个模型，均已上传至 ModelScope 平台：

1.  **[mc36473/Qwen3-4B-ChnSentiCorp-Sentiment](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-Sentiment/summary)** (基于 **清洗后** 数据训练) - **推荐使用**
2.  **[mc36473/Qwen3-4B-ChnSentiCorp-ACC97](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-ACC97/summary)** (基于原始带噪数据训练)

## 环境安装与准备

请遵循以下步骤来配置运行推理和训练所需的环境。

### 步骤 1: 克隆本项目

```bash
git clone https://github.com/your-username/Qwen3-ChnSentiCorp-Cleaning-Experiment.git
cd Qwen3-ChnSentiCorp-Cleaning-Experiment
```

### 步骤 2: 安装 LLaMA-Factory

本项目依赖 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 进行模型训练和 API 服务部署。

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# 安装时加入对 PyTorch、评估指标和 vLLM 推理引擎的支持
pip install -e ".[torch,metrics,vllm]"
cd ..
```
*注意：请参考 LLaMA-Factory 官方文档以满足详细的环境要求（例如 CUDA 版本等）。*

### 步骤 3: 安装评测脚本依赖

```bash
pip install -r requirements.txt
```

## 如何使用（推理与评估）

完成环境安装后，您可以轻松地启动模型并进行性能评估。

### 步骤 1: 启动 API 服务

运行脚本来启动推理服务。它将自动从 ModelScope 下载推荐的模型，并通过 vLLM 提供 API 接口。

```bash
bash scripts/run_api_server.sh
```
服务启动后，可在 `http://localhost:8000` 访问。

### 步骤 2: 运行评估

服务运行后，执行评估脚本。

```bash
# 进入 scripts 目录执行
cd scripts/
python evaluate.py
```
评测结果将直接打印在控制台，详细的预测记录会保存在 `scripts/evaluation_results.csv` 文件中。

## 如何复现训练

在开始训练前，请确保您已完成 **环境安装与准备** 部分的所有步骤。训练脚本将从基座模型 `Qwen/Qwen3-4B-Instruct-2507` 开始微调。

### 执行训练

-   **训练（清洗后数据）:**
    ```bash
    # 在本项目的根目录下执行
    bash scripts/train_cleaned.sh
    ```
-   **训练（原始带噪数据）:**
    ```bash
    # 在本项目的根目录下执行
    bash scripts/train_noisy.sh
    ```

## 仓库结构

```
.
├── data/
│   ├── issues/         # 标签噪声问题样例
│   ├── training/       # 训练集（带噪与清洗后）
│   └── test/           # 测试集
├── scripts/
│   ├── train_*.sh      # 训练脚本
│   ├── run_api_server.sh # 启动推理服务脚本
│   └── evaluate.py     # 评估脚本
├── config/
│   └── dataset_info.json # LlamaFactory 数据集配置文件
└── ...