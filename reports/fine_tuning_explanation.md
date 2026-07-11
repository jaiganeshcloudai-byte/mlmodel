# Fine-Tuning Methodology Explanation Report

**File Path Location:** `reports/fine_tuning_explanation.md`  
**Objective:** Document the theoretical concepts of Parameter-Efficient Fine-Tuning (PEFT) and state the exact hyperparameter configurations utilized across the 3-stage customer support training workflow.

---

### 1. Conceptual Framework Review

* **Full Fine-Tuning Expense:** Modifying all parameters requires storing massive gradient and optimizer tracking metrics, demanding expensive multi-GPU cluster arrays.
* **LoRA Framework:** Freezes the foundational model layers and appends small, trainable low-rank decomposition matrices ($r$), lowering parameter training states by over 99%.
* **QLoRA Framework:** Quantizes the base frozen weights into a specialized 4-bit NormalFloat (NF4) layout, lowering the VRAM baseline memory foot-print to allow fine-tuning on a single consumer GPU (like a Tesla T4).
* **Non-Instruction Fine-Tuning:** Unstructured causal language modeling to absorb raw domain vocabulary, terminology, and background knowledge.
* **Instruction Fine-Tuning (SFT):** Alignment phase mapping domain data into structured Question/Response prompt wrappers.
* **DPO Alignment:** Binary preference margin optimization utilizing relative contrastive pairs (Chosen vs. Rejected) to enforce tone safety and prune hallucinations.

---

### 2. Configured Project Hyperparameters

The following specific hyperparameter bounds were consistently utilized across our Unsloth-accelerated training pipeline to prevent memory failure and maximize optimization stability:

| Hyperparameter | Configured Value | Engineering Justification |
| :--- | :--- | :--- |
| **Rank ($r$)** | `16` | Chosen to balance capacity and VRAM overhead. A rank of 16 provides adequate parameter freedom to capture specialized customer service alignment constraints without bloating memory. |
| **Alpha ($\alpha$)** | `32` | Scaled at a stable 2:1 ratio relative to rank ($r=16$). This scales the adapter weight updates appropriately, ensuring instruction format layout rules override general base text patterns without causing optimization destabilization. |
| **Dropout** | `0` | Explicitly configured to 0. Unsloth's native fast kernel layers are mathematically optimized for zero-dropout conditions, maximizing throughput velocity and stabilizing lower-rank computations. |
| **Learning Rate** | `2e-4` (Stage 1)<br>`2e-5` (Stage 2 SFT)<br>`5e-6` (Stage 3 DPO) | Progressively scaled down stage by stage. Stage 1 utilizes a standard rate for text absorption, while Stage 2 and Stage 3 use highly conservative rates to safely shift preference boundaries without destroying previously learned background domain data. |
| **Batch Size** | `Per-device: 2`<br>`Grad Accumulation: 4`<br>`Effective: 8` | Configured to fit cleanly inside the 15GB VRAM capacity of a single Tesla T4 GPU. Using an effective batch size of 8 via Gradient Accumulation optimizes the optimization vector steps while preventing out-of-memory errors. |

---

### 3. Conclusion & Synthesis
By integrating LoRA, QLoRA, and Unsloth optimizations, this 3-stage pipeline efficiently transforms a lightweight base model into an interview-ready, production-grade customer support assistant. The progressive architectural alignment shifts the model from raw vocabulary mastery (Stage 1), to strict instruction following (Stage 2), and finally to elite preference compliance (Stage 3)—all achieved completely on free, accessible consumer hardware.
