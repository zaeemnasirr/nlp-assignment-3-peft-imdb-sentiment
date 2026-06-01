# NLP Assignment 3: PEFT Methods for IMDb Sentiment Classification

This repository contains the implementation report and experiment results for NLP Assignment 3.

## Project Overview

This project compares full model fine-tuning and parameter-efficient fine-tuning methods for IMDb binary sentiment classification.

The methods compared are:

1. Full Fine-Tuning
2. LoRA
3. QLoRA
4. IA3

The comparison uses accuracy, F1-macro, training time, peak GPU memory, and trainable parameters.

## Dataset

The project uses the IMDb sentiment classification dataset from Hugging Face.

Each example is classified into one of two sentiment classes:

- Positive
- Negative

## Models Used

The main model families used in the experiments are:

- RoBERTa-base
- DistilBERT
- DeBERTa-v3

The final valid comparison mainly uses RoBERTa-base and DistilBERT because DeBERTa-v3 logs were incomplete or unavailable for final comparison.

## Methods Compared

### Full Fine-Tuning

Full Fine-Tuning updates all pretrained transformer weights and the classification head. It gives the strongest accuracy but requires the largest number of trainable parameters.

### LoRA

LoRA freezes the base model and trains low-rank adapter matrices. It reduces trainable parameters while keeping strong accuracy.

### QLoRA

QLoRA uses a quantized base model with LoRA adapters. It gives the best memory-efficiency trade-off in this project.

### IA3

IA3 uses lightweight multiplicative adapter vectors. It is fast and parameter-efficient but less accurate than LoRA and QLoRA in this experiment.

## Best Valid Configuration per Method

| Method | Model | Configuration | LR | Accuracy | F1 | Time | VRAM | Trainable Params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Full Fine-Tuning | RoBERTa-base | Full Fine-Tuning | 2e-5 | 93.85 ± 0.04 | 93.85 ± 0.04 | 468.8 ± 9.5s | 3.678 GB | 124.647M |
| LoRA | RoBERTa-base | r=16, Attention+FFN | 2e-5 | 91.97 ± 0.07 | 91.96 ± 0.07 | 546.7 ± 18.1s | 2.644 GB | 2.951M |
| QLoRA | RoBERTa-base | NF4, r=16, double quantization | 2e-5 | 89.54 ± 0.13 | 89.54 ± 0.13 | 715.6 ± 4.5s | 0.626 GB | 1.182M |
| IA3 | DistilBERT | Attention+FFN | 2e-5 | 80.67 ± 0.01 | 80.67 ± 0.01 | 221.4 ± 0.9s | 2.644 GB | 0.620M |

## Key Findings

- Full Fine-Tuning achieved the highest accuracy.
- LoRA was the best PEFT method in terms of accuracy.
- QLoRA gave the best memory-efficiency trade-off.
- IA3 was lightweight and fast but less accurate than LoRA and QLoRA.
- Increasing learning rate from 1e-5 to 2e-5 improved the best PEFT configurations.
- Attention+FFN placement performed better than attention-only placement for LoRA.

## Repository Structure

```text
.
├── README.md
├── NLP_Assignment_3_Report.pdf
├── notebooks/
│   ├── 1-full-fine-tuning.ipynb
│   ├── 2-3-4-lora_all_experiments.ipynb
│   ├── 5-qlora.ipynb
│   ├── 6-ia3-adapter.ipynb
│   └── exp07-lr-2e-5_lora_qlora_adapter.ipynb
├── results/
│   ├── 1-full_ft_experiment_results.csv
│   ├── 2-3-4-lora_experiment_results.csv
│   ├── 5-qlora_experiment_results.csv
│   ├── ia3_adapter_experiment_results.csv
│   └── exp07_full_ft_experiment_results.csv
└── images/
    ├── best_accuracy_comparison.png
    ├── pareto_vram_accuracy.png
    ├── pareto_time_accuracy.png
    ├── lora_rank_sensitivity.png
    ├── module_placement_ablation.png
    └── accuracy_vs_trainable_parameters.png
