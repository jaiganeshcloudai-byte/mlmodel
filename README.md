# Domain-Specific AI Assistant: Customer Support AI

This repository contains the complete 3-stage fine-tuning implementation for an industry-style domain-specific AI Customer Support Assistant. Leveraging **Unsloth**, **LoRA/QLoRA**, and **Direct Preference Optimization (DPO)**, this project demonstrates how to adapt an open-source Large Language Model (LLM) to master internal company policies, follow complex instruction templates, and prioritize safe, professional customer resolutions.

---

## 🛠️ The 3-Stage Fine-Tuning Workflow
1. **Stage 1: Non-Instruction Fine-Tuning (`notebooks/non_instruction_finetuning.ipynb`)**: Domain adaptation via raw text absorption.
2. **Stage 2: Supervised Fine-Tuning / SFT (`notebooks/instruction_finetuning.ipynb`)**: Turning a text completer into an interactive assistant.
3. **Stage 3: Direct Preference Optimization / DPO (`notebooks/dpo_alignment.ipynb`)**: Aligning responses with corporate safety, clarity, and tone guidelines.

## 📁 Repository Structure
```text
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   └── final_evaluation.md
├── src/
│   └── inference.py
├── .gitignore
└── README.md
'''
