# Attention Is All You Need — Built From Scratch, Annotated

A pedagogical, end-to-end re-implementation of the Transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)) in **plain PyTorch** — every component built by hand, heavily annotated, and verified by actually training the model and looking inside its attention heads.

> Built for understanding, not for reuse: nothing is hidden behind `nn.Transformer`. Each block has a theory cell explaining *why* the code is shaped the way it is, followed by runnable code whose output is captured in the notebook.

## What's inside

The notebook [`attention_from_scratch.ipynb`](attention_from_scratch.ipynb) builds and verifies, in order:

| Paper section | Component | What's demonstrated |
|---|---|---|
| §3.5 | Sinusoidal positional encoding | Numerically **proves** the linear-offset property (`PE[pos+k]` is a fixed linear map of `PE[pos]`) |
| §3.2.1 | Scaled dot-product attention | Shows `Var(q·k) = dₖ` (footnote 4) and softmax saturation → why we divide by `√dₖ` |
| §3.2.2 | Multi-head attention (h=8) | Fused-projection implementation |
| §3.3 / §3.1 | Position-wise FFN + Add & Norm | Faithful Post-LN |
| §3.2.3 | Padding + causal masking | Visualized |
| §3.4 | Embeddings (×√d_model) + generator | Measures the scaling effect on token vs. position signal |
| §3.1 | Encoder & Decoder stacks | Full encoder–decoder assembly (~930K params) |
| §5.3 | Adam + Noam warmup schedule | Plotted (warmup → inverse-sqrt decay) |
| §5.4 | Dropout + label smoothing | KL-divergence formulation, visualized |
| §6.1 | Greedy autoregressive decoding | **5/5 exact copies** on unseen sequences |
| Figs 3–5 | Attention visualization | Cross-attention heads show the diagonal "copy" pattern |

The model is trained on a synthetic **copy task** (target = source) — small enough to converge in seconds, but it genuinely exercises encoder self-attention, masked decoder self-attention, and encoder–decoder cross-attention.

## Run it

Open in Google Colab (a free CPU runtime is plenty; a T4 GPU trains in ~10s):

1. Upload `attention_from_scratch.ipynb` to [Colab](https://colab.research.google.com/) (or open from GitHub: **File → Open notebook → GitHub**).
2. **Runtime → Run all.** If the runtime ever disconnects, just re-run the **Foundations** and **Config** cells and continue.

Requirements: `torch`, `matplotlib` (both preinstalled on Colab).

## License

MIT — see [LICENSE](LICENSE).

---
*Built as a learning project. The annotations include honest mid-build corrections (e.g. fixing a numerically unstable proof and an incorrect embedding-norm claim) — debugging in the open is part of the pedagogy.*
