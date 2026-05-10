# 🦙 End-to-End Llama 4 Mixture-of-Experts (MoE) Text Model — From Scratch in PyTorch

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
  <img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA"/>
</p>

<p align="center">
  A detailed, step-by-step educational implementation of a Llama 4-inspired <strong>Mixture-of-Experts (MoE)</strong> language model built entirely from scratch in PyTorch — without relying on any high-level model libraries.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#%EF%B8%8F-architecture)
- [Hyperparameters](#%EF%B8%8F-hyperparameters)
- [Notebook Structure](#-notebook-structure)
- [Getting Started](#-getting-started)
- [Training Results](#-training-results)
- [Key Concepts Explained](#-key-concepts-explained)
- [Why Llama 4 Stands Apart from Other LLMs](#-why-llama-4-stands-apart-from-other-llms)
- [Simplifications vs. Real Llama 4](#-simplifications-vs-real-llama-4)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 📖 Overview

This Jupyter notebook walks through building a simplified but architecturally faithful text generation model inspired by **Meta's Llama 4** architecture, with a core focus on the **Mixture-of-Experts (MoE)** mechanism. Every component is implemented inline with detailed theoretical explanations, shape annotations, and print-based debugging — making it ideal for learners and researchers who want to deeply understand modern LLM internals.

The project is designed to be **read like a textbook** — each code block is preceded by theory, and followed by an output explanation.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **MoE Architecture** | Full Top-K expert routing with weighted combination and a shared expert |
| **RMSNorm** | Root Mean Square Layer Normalization (no learnable bias) |
| **RoPE** | Rotary Positional Embeddings applied inline to Q and K vectors |
| **Multi-Head Attention** | Causal self-attention with precomputed masks |
| **Gated MLP Experts** | SiLU-gated feed-forward networks inside each expert |
| **AdamW Optimizer** | Standard Transformer training optimizer |
| **Character-Level Tokenizer** | Simple vocab built from training corpus |
| **Autoregressive Generation** | Temperature-sampled next-token prediction |

---

## 🏗️ Architecture

The model is a decoder-only Transformer with MoE layers replacing standard FFN blocks:

```
Input Tokens
     │
     ▼
Token Embedding  (vocab_size → d_model)
     │
     ▼
 ┌─────────────────────────────────┐
 │  Transformer Block × n_layers  │
 │                                 │
 │   ┌─────────────────────────┐   │
 │   │  RMSNorm                │   │
 │   │  Multi-Head Attention   │   │
 │   │  + RoPE on Q & K        │   │
 │   │  Causal Mask            │   │
 │   └─────────────────────────┘   │
 │          + Residual             │
 │   ┌─────────────────────────┐   │
 │   │  RMSNorm                │   │
 │   │  MoE Block:             │   │
 │   │    Router → Top-K       │   │
 │   │    Expert MLPs (×N)     │   │
 │   │    Shared Expert MLP    │   │
 │   │    Combine & Add        │   │
 │   └─────────────────────────┘   │
 │          + Residual             │
 └─────────────────────────────────┘
     │
     ▼
Final RMSNorm
     │
     ▼
Output Linear  (d_model → vocab_size)
     │
     ▼
   Logits
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 128 | Embedding / hidden dimension |
| `n_layers` | 4 | Number of Transformer blocks |
| `n_heads` | 4 | Number of attention heads |
| `d_k` | 32 | Dimension per attention head |
| `block_size` | 64 | Maximum context length |
| `num_local_experts` | 4 | Number of MoE experts per layer |
| `num_experts_per_tok` | 2 | Top-K experts selected per token |
| `expert_dim` | 256 | Hidden size of each expert MLP |
| `shared_expert_dim` | 256 | Hidden size of the shared expert MLP |
| `rope_theta` | 10,000 | RoPE base frequency |
| `rms_norm_eps` | 1e-5 | RMSNorm stability epsilon |
| `learning_rate` | 5e-4 | AdamW learning rate |
| `batch_size` | 16 | Training batch size |
| `epochs` | 3,000 | Total training iterations |

> **Total Trainable Parameters:** ~2,240,640

---

## 📚 Notebook Structure

The notebook is organized into 7 sequential steps:

```
Step 0 — Environment Setup (PyTorch, seed, device)
Step 1 — Corpus & Character-Level Tokenization
  ├── 1.1 Define Training Corpus
  ├── 1.2 Build Vocabulary (char ↔ int mappings)
  └── 1.3 Encode Corpus to Tensor
Step 2 — Hyperparameter Definition
Step 3 — Data Preparation
  ├── 3.1 Create Input/Target Pairs (next-token prediction)
  └── 3.2 Random Batch Sampling Strategy
Step 4 — Model Component Initialization
  ├── 4.1 Token Embedding Layer
  ├── 4.2 RoPE Inverse Frequency Precomputation
  ├── 4.3 RMSNorm Weight Parameters
  ├── 4.4 Multi-Head Attention Linear Layers
  ├── 4.5 MoE Components (Router, Experts, Shared Expert)
  ├── 4.6 Final Output Linear Layer
  └── 4.7 Causal Attention Mask
Step 5 — Training Setup
  ├── 5.1 AdamW Optimizer
  └── 5.2 Cross-Entropy Loss Function
Step 6 — Training Loop (Forward Pass + Backprop)
Step 7 — Text Generation (Autoregressive Inference)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ (with CUDA support recommended)
- Jupyter Notebook or JupyterLab

### Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install jupyter matplotlib
```

### Run the Notebook

```bash
jupyter notebook code.ipynb
```

> ✅ The notebook auto-detects GPU (`cuda`) or falls back to CPU. For the 3,000-epoch run, **GPU is strongly recommended**.

---

## 📊 Training Results

The model was trained for **3,000 epochs** on a small character-level corpus (Lewis Carroll's *Alice's Adventures in Wonderland* excerpt). Loss converges from ~3.81 → ~0.055:

```
Epoch    1/3000   |  Loss: 3.8124
Epoch  301/3000   |  Loss: 0.0734
Epoch  601/3000   |  Loss: 0.0595
Epoch  901/3000   |  Loss: 0.0609
Epoch 1201/3000   |  Loss: 0.0707
Epoch 1501/3000   |  Loss: 0.0664
Epoch 1801/3000   |  Loss: 0.0559
Epoch 2101/3000   |  Loss: 0.0610
Epoch 2401/3000   |  Loss: 0.0680
Epoch 2701/3000   |  Loss: 0.0641
Epoch 3000/3000   |  Loss: 0.0553
```

A loss curve is automatically plotted using Matplotlib at the end of training.

---

## 🧠 Key Concepts Explained

### Mixture-of-Experts (MoE)
Instead of a single FFN, each MoE block contains **N independent expert MLPs** and a **router** network. The router selects the top-K experts for each token and linearly combines their outputs. A **shared expert** (always active) ensures stable base representations.

### Rotary Positional Embeddings (RoPE)
Rather than adding positional encodings to token embeddings, RoPE **rotates** the query and key vectors in complex space based on each token's absolute position. This allows the model to capture relative position information naturally in the attention scores.

### RMSNorm
A simplified alternative to LayerNorm that normalizes by the root mean square of activations — removing the need for mean subtraction and a learnable bias term, improving training stability and speed.

---

## 🦙 Why Llama 4 Stands Apart from Other LLMs

Meta's **Llama 4** (released April 2025) is not just an iteration — it is a **fundamental architectural rethink** that sets it apart from both its predecessors (Llama 1/2/3) and competing frontier models (GPT-4o, Claude 3.5, Gemini 2.0). Here is a deep dive into what makes it genuinely unique.

---

### 1. 🔀 Native Mixture-of-Experts (MoE) — Sparse Intelligence at Scale

Llama 1/2/3 were **dense models**: every parameter was activated for every token. Llama 4 shifts to a **sparse MoE architecture**, which is one of the most significant efficiency breakthroughs in modern LLMs.

**How it works:**
- Each transformer block replaces the standard FFN with **N independent Expert MLPs** + a **Shared Expert** (always active).
- A learned **Router** network selects only the **Top-K experts** (e.g., 2 out of 16) for each token.
- The outputs of the selected experts are **weighted and summed**.

**Why it matters vs. other LLMs:**

| Model | Architecture | Active Params / Token | Total Params |
|:---|:---|:---:|:---:|
| GPT-4o | Dense (estimated) | ~100% | ~200B (est.) |
| Claude 3.5 Sonnet | Dense | ~100% | Undisclosed |
| Llama 3.1 405B | Dense | 100% | 405B |
| **Llama 4 Scout** | **MoE** | **~16%** (17B / 109B) | **109B** |
| **Llama 4 Maverick** | **MoE** | **~4%** (17B / 400B) | **400B** |

> **Key Insight:** Llama 4 Maverick activates only **17B parameters per token** despite having 400B total — giving it the *reasoning quality of a 400B model* at the *inference cost of a ~17B model*.

---

### 2. 🌀 iRoPE — Interleaved RoPE for Infinite Context

Standard RoPE (used in Llama 1/2/3, Qwen, Mistral) applies positional rotations to **every** attention layer, which limits practical context length and degrades at extrapolation.

Llama 4 introduces **iRoPE (Interleaved Rotary Position Embeddings)**:

```
Layer 1  →  Standard RoPE attention      (position-aware)
Layer 2  →  No-RoPE attention            (position-free, global)
Layer 3  →  Standard RoPE attention
Layer 4  →  No-RoPE attention
   ...
```

- **Position-aware layers** handle local syntactic patterns precisely.
- **Position-free layers** capture global, long-range semantic relationships without positional bias.
- Together they unlock a **10,000,000 token context window** (Scout model) — compared to 128K for GPT-4o and 200K for Claude 3.5.

| Model | Max Context Window |
|:---|:---:|
| GPT-4o | 128,000 tokens |
| Claude 3.5 Sonnet | 200,000 tokens |
| Gemini 2.0 Flash | 1,000,000 tokens |
| Llama 3.1 | 128,000 tokens |
| **Llama 4 Maverick** | **1,000,000 tokens** |
| **Llama 4 Scout** | **10,000,000 tokens** 🏆 |

---

### 3. 🎨 Native Multimodality — Vision Baked In from Day One

Unlike Llama 3.2 Vision (which bolted vision adapters onto a text model), Llama 4 was designed from the ground up with **joint text + image + video understanding**:

- Uses **MetaCLIP** as the vision encoder — Meta's own vision-language contrastive encoder.
- Employs **Early Fusion**: visual tokens are mixed with text tokens at the embedding level before passing through transformer layers.
- This is in contrast to **Late Fusion** (used by GPT-4V, LLaVA, etc.) where vision features are projected and injected later.

```
Early Fusion (Llama 4)          Late Fusion (Most Others)
────────────────────            ───────────────────────────
[Image Tokens]                  Transformer Layers
[Text  Tokens]  ──→ Transformer     ↑
     mixed at                   [Projected Image Features]
     input level                [Text Tokens only at start]
```

> **Benefit:** Early fusion allows every transformer layer to jointly reason over both modalities — leading to richer cross-modal understanding compared to late-fusion approaches.

---

### 4. 🔓 Open Weights — The Transparency Advantage

All three Llama 4 variants are **open-weight** models — a major differentiator from every proprietary frontier model:

| Property | Llama 4 | GPT-4o | Claude 3.5 | Gemini 2.0 |
|:---|:---:|:---:|:---:|:---:|
| Weights publicly available | ✅ | ❌ | ❌ | ❌ |
| Local / on-premise deployment | ✅ | ❌ | ❌ | ❌ |
| Fine-tuning allowed | ✅ | Limited | ❌ | ❌ |
| Full architectural transparency | ✅ | ❌ | ❌ | ❌ |
| Data privacy (no API calls) | ✅ | ❌ | ❌ | ❌ |

> This makes Llama 4 the model of choice for **regulated industries** (healthcare, finance, legal), **air-gapped environments**, and anyone who needs full control over their AI stack.

---

### 5. 📐 Grouped Query Attention (GQA) + L2-Norm on Q/K

Real Llama 4 also uses:
- **GQA**: Multiple query heads share a single K/V pair — reducing memory during inference while maintaining quality (this project simplifies to standard MHA for clarity).
- **L2 Norm on Q and K**: Normalizes query and key vectors before dot-product attention, preventing logit explosion in very long sequences — critical for the 10M context window.

---

### 6. 🧠 The Llama 4 Model Family at a Glance

| Model | Active Params | Total Params | Context | Use Case |
|:---|:---:|:---:|:---:|:---|
| **Llama 4 Scout** | 17B | 109B | 10M tokens | Long-context; runs on 1× H100 |
| **Llama 4 Maverick** | 17B | 400B | 1M tokens | General-purpose; best quality/cost |
| **Llama 4 Behemoth** | 288B | ~2T | — | Teacher model; frontier research |

> **Llama 4 Behemoth** (2T parameters, 288B active) is used as a **teacher model** for knowledge distillation — training Scout and Maverick to punch far above their active-parameter weight class.

---

### 7. ⚡ Summary: Llama 4 vs. the LLM Landscape

| Innovation | Llama 4 | Earlier Llamas | GPT / Claude / Gemini |
|:---|:---:|:---:|:---:|
| Sparse MoE (efficiency) | ✅ | ❌ Dense | Partial / Proprietary |
| iRoPE (10M+ context) | ✅ | ❌ | Limited |
| Native multimodality (early fusion) | ✅ | Partial (adapter-based) | Partial |
| Open weights | ✅ | ✅ | ❌ |
| GQA + Q/K normalization | ✅ | Partial | Unknown |
| Fine-grained expert routing | ✅ | ❌ | Unknown |
| On-device / private deployment | ✅ | ✅ | ❌ |

> **Bottom line:** Llama 4 is the first open-weight model family to combine **sparse MoE efficiency**, **industry-leading context length**, and **native multimodality** in a single architecture — making it one of the most architecturally ambitious open-source LLMs ever released.

---

## 🔬 Simplifications vs. Real Llama 4

| Feature | This Implementation | Real Llama 4 |
|---|---|---|
| Attention type | Standard MHA | Grouped Query Attention (GQA) |
| Vocabulary | 36 chars (character-level) | ~128K tokens (BPE) |
| Context length | 64 tokens | 10M+ tokens |
| Number of experts | 4 | 16+ |
| Training data | ~600 chars | Trillions of tokens |
| RoPE theta | 10,000 | 500,000 |
| L2 Norm on Q/K | ❌ | ✅ |
| Chunked attention | ❌ | ✅ |

---

## 📂 Project Structure

```
📦 End-to-End Llama 4 MoE Text Model — From Scratch in PyTorch/
 ┣ 📓 code.ipynb     # Main implementation notebook
 ┗ 📄 README.md      # Project documentation (this file)
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Inspired by **Meta's Llama 4** architecture and technical reports
- Character-level language model concept from **Andrej Karpathy's** nanoGPT
- Dataset excerpt: *Alice's Adventures in Wonderland* by Lewis Carroll (public domain)
