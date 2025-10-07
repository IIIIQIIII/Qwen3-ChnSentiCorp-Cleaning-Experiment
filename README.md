# Improving ChnSentiCorp Sentiment Analysis by Cleaning Label Noise

[![ModelScope](https://img.shields.io/badge/ModelScope-Qwen3--4B--ChnSentiCorp--Sentiment-blue)](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-Sentiment/summary)
[![ModelScope](https://img.shields.io/badge/ModelScope-Qwen3--4B--ChnSentiCorp--ACC97-gray)](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-ACC97/summary)

[ English | [中文](README_zh.md) ]

---

This repository documents an experiment to improve the performance of a sentiment analysis model on the ChnSentiCorp dataset by cleaning its label noise. We fine-tuned the **Qwen3-4B** model using LLaMA-Factory on both the original (noisy) and the cleaned versions of the dataset.

The results clearly demonstrate that training on the cleaned dataset significantly boosts the model's accuracy and F1-score.

## Key Findings

Training on the cleaned dataset resulted in a **1.67%** absolute improvement in accuracy on the test set.

| Model Trained On | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| :--------------- | :------: | :---------------: | :------------: | :--------------: |
| Original (Noisy) Data |  97.21%  |      97.14%       |     97.31%     |      97.20%      |
| **Cleaned Data** | **98.88%** |    **98.88%**     |   **98.88%**   |    **98.88%**    |

## Models

Two models were fine-tuned and are available on ModelScope:

1.  **[Qwen3-4B-ChnSentiCorp-Sentiment](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-Sentiment/summary)** (Trained on **Cleaned Data**) - **Recommended**
2.  **[Qwen3-4B-ChnSentiCorp-ACC97](https://modelscope.cn/models/mc36473/Qwen3-4B-ChnSentiCorp-ACC97/summary)** (Trained on Original Noisy Data)

## Setup and Installation

Follow these steps to set up the environment for both inference and training.

### Step 1: Clone This Repository

```bash
git clone https://github.com/IIIIQIIII/Qwen3-ChnSentiCorp-Cleaning-Experiment.git
cd Qwen3-ChnSentiCorp-Cleaning-Experiment
```

### Step 2: Install LLaMA-Factory

This project relies on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training and API serving.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# Install with support for PyTorch, evaluation metrics, and vLLM for inference
pip install -e ".[torch,metrics,vllm]"
cd ..
```
*Note: Please refer to the official LLaMA-Factory documentation for detailed environment requirements (e.g., CUDA version).*

### Step 3: Install Evaluation Script Dependencies

```bash
pip install -r requirements.txt
```

## How to Use (Inference and Evaluation)

After completing the setup, you can easily run inference and evaluate the model.

### Step 1: Start the API Server

Launch the inference service. It will download the recommended model from ModelScope and serve it via a vLLM-powered API endpoint.

```bash
bash scripts/run_api_server.sh
```
The server will be available at `http://localhost:8000`.

### Step 2: Run Evaluation

Once the server is running, execute the evaluation script.

```bash
# Navigate to the scripts directory to run
cd scripts/
python evaluate.py
```
The results will be printed to the console and saved to `scripts/evaluation_results.csv`.

## How to Reproduce Training

Ensure you have completed the **Setup and Installation** section before proceeding. The training scripts start from the base model `Qwen/Qwen3-4B-Instruct-2507`.

### Run Training

-   **Train on Cleaned Data:**
    ```bash
    # From the root directory of this project
    bash scripts/train_cleaned.sh
    ```
-   **Train on Noisy Data:**
    ```bash
    # From the root directory of this project
    bash scripts/train_noisy.sh
    ```

## Repository Structure

```
.
├── data/
│   ├── issues/         # Examples of label noise
│   ├── training/       # Training sets (noisy and cleaned)
│   └── test/           # Test set
├── scripts/
│   ├── train_*.sh      # Training scripts
│   ├── run_api_server.sh # Script to start inference server
│   └── evaluate.py     # Evaluation script
├── config/
│   └── dataset_info.json # Dataset config for LLaMA-Factory
└── ...
```