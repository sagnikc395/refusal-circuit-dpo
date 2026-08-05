# refusal-circuit-dpo

## objective 
in a 500M-parameter base model, which layers , attention heads, and residual stream directions are responsible for the emergent refusal behavior introduced by DPO

## methodology

Here is a concrete, end-to-end plan for **Project 1**, scoped specifically for your M4 MacBook, your mech interp background, and a 4-week timeline.

---

## Project Title
**"Localizing the Refusal Circuit: A Mechanistic Analysis of DPO-Induced Safety Behaviors in Sub-1B LLMs"**

## The Core Pitch
You will take a 500M-parameter base model, run it through a full post-training pipeline (SFT → DPO), and then use mechanistic interpretability to answer a real research question: **"Which layers, attention heads, and residual stream directions are responsible for the emergent refusal behavior introduced by DPO?"**

This bridges your existing safety + interp expertise with the modern post-training stack. It is not "yet another RLHF demo." It is an investigation into *how alignment algorithms mechanistically rewrite model behavior*.

---

## 1. Exact Model & Compute Budget

| Component | Specification |
|-----------|---------------|
| **Base Model** | `Qwen/Qwen2.5-0.5B` (base, not instruct) |
| **Reference** | `Qwen/Qwen2.5-0.5B-Instruct` (for sanity checks only) |
| **Hardware** | MacBook Air M4, 16GB unified memory |
| **Backend** | `torch.bfloat16` on `mps` |
| **PEFT** | LoRA, rank 16, targeting all linear attention + MLP layers |
| **Training Memory** | ~6-8GB peak (well within 16GB) |
| **Storage** | ~5GB for models, datasets, and cached activations |

---

## 2. The Research Question

> **"In a 0.5B model aligned via DPO, is refusal computed early (token identification) or late (output suppression)? Can we localize it to specific middle-layer components, and can we steer the model into or out of refusal by intervening on the residual stream?"**

This mirrors the spirit of Anthropic's refusal work but is scoped to DPO on a tiny model—something you can actually run locally and still get publishable-quality insights.

---

## 3. The Pipeline (Step-by-Step)

### Stage A: Supervised Fine-Tuning (SFT)
**Goal:** Teach the model to follow instructions. Explicitly **avoid** teaching it safety refusals.

- **Dataset:** `yahma/alpaca-cleaned`. Sample 1,000 diverse, benign instructions. Filter out any examples that touch on safety, ethics, or refusals. You want a helpful but "naively compliant" model.
- **Format:** Use a simple instruction template (e.g., `### Instruction:\n{prompt}\n\n### Response:\n{response}`). Do not use the chat template yet; you want the base model to learn the format, not safety norms.
- **Training:** 1 epoch, batch size 1, gradient accumulation steps 8 (effective batch 8), learning rate 5e-5, max seq length 512.
- **Output:** `sft-model` (naively helpful, will comply with harmful requests).

### Stage B: Direct Preference Optimization (DPO)
**Goal:** Induce refusal behavior on harmful prompts while preserving helpfulness on benign prompts.

- **Dataset:** `HuggingFaceH4/hh-rlhf`. You will curate a small, high-quality subset locally:
  1. Load the full dataset.
  2. Filter for examples where `chosen` starts with refusal keywords (`"I cannot"`, `"I'm sorry"`, `"I can't help"`, etc.) and `rejected` does not.
  3. Extract the user turn as the `prompt`, and the first assistant turn as `chosen`/`rejected`.
  4. Keep **400 refusal pairs** + **100 general helpfulness pairs** (where `chosen` is just better than `rejected`, not refusal-related). This prevents DPO from making the model universally unhelpful.
  5. Truncate all sequences to 512 tokens.
- **Training:** 1 epoch, batch size 1, gradient accumulation 8, learning rate 5e-6, β=0.1. Use your SFT model as the reference model (`model_ref`).
- **Output:** `dpo-model` (should now refuse harmful requests).

### Stage C: Evaluation Harness
Before interp, verify the pipeline worked. Build a tiny evaluation script:

- **Harmful set:** 50 prompts from the HH-RLHF red-teaming distribution.
- **Benign set:** 50 prompts from Alpaca (held-out).
- **Metric:** Refusal rate on harmful (should be >80% for DPO, <20% for SFT) and answer rate on benign (should be >90% for both).
- **Baseline:** Run the same eval on `Qwen2.5-0.5B-Instruct` to show your DPO model approaches official alignment quality.

---

## 4. The Mech Interp Methodology (The Heart of the Project)

You will run three core experiments. All experiments operate on **next-token prediction** (not full generation) because it is cleaner, faster, and standard for circuit analysis.

### Experiment 1: Logit Lens (Where do refusal tokens emerge?)
- Select 20 harmful prompts from your eval set.
- For each model (Base, SFT, DPO), run a forward pass.
- At **every layer** (0 to 23), take the residual stream, apply the final LayerNorm + unembedding matrix, and check the probability of top refusal tokens (`"I"`, `"Sorry"`, `"Cannot"`).
- **Deliverable:** A line plot: Layer (x-axis) vs. Refusal Token Probability (y-axis), with three lines (Base, SFT, DPO).
- **Hypothesis:** DPO will show a sharp spike in late-middle layers (12–18), while Base/SFT will remain flat.

### Experiment 2: Activation Patching (Which layers are causal?)
- **Clean run:** Run the DPO model on a harmful prompt. It refuses. Save all intermediate activations.
- **Corrupted run:** Run the SFT model on the same prompt. It complies. Save activations.
- **Intervention:** For each layer `L` (0 to 23), run the DPO model but **patch in the SFT model's residual stream** at layer `L` (using forward hooks). Measure whether the output flips from refusal to compliance.
- **Variation:** Do this separately for **attention output** and **MLP output** to see if refusal is computed in attention heads or MLPs.
- **Deliverable:** A heatmap showing "refusal disruption score" per layer and component.
- **Hypothesis:** You will find a concentrated band of 2–4 layers where patching breaks refusal. This is your "refusal circuit."

### Experiment 3: Steering Vectors (Can we control refusal?)
- Identify the critical layer `L*` from Experiment 2 (e.g., layer 14).
- Compute two mean activation vectors at `L*`:
  - **μ_refuse:** Mean residual stream over 20 harmful prompts run through DPO model.
  - **μ_comply:** Mean residual stream over 20 benign prompts run through DPO model.
- **Steering vector:** `v = μ_refuse - μ_comply`.
- **Test:** Take the **SFT model** (which normally complies), add `α * v` to layer `L*` during inference on harmful prompts, and measure refusal probability for different scalars `α` (e.g., 0, 0.5, 1.0, 2.0, 5.0).
- **Reverse test:** Take the **DPO model**, subtract `α * v`, and see if it complies.
- **Deliverable:** A plot of refusal probability vs. steering coefficient `α`.
- **Impact:** This proves you haven't just *found* the circuit—you can *control* it.

### Stretch Experiment (Week 4, if time permits): Linear Probing
- Train a simple logistic regression probe on layer `L*` activations to predict "will the next token be a refusal?"
- Show that probe accuracy increases with layer depth, confirming that the model computes refusal late.

---

## 5. Deliverables & Repo Structure

Your GitHub repo should look like this:

```
refusal-circuit-dpo/
├── README.md                 # Research summary with key figures
├── requirements.txt
├── data/
│   ├── build_sft_dataset.py
│   ├── build_dpo_dataset.py
│   └── prompts/              # Your 50 harmful + 50 benign eval prompts
├── training/
│   ├── train_sft.py
│   ├── train_dpo.py
│   └── configs/
├── evaluation/
│   └── eval_refusal.py
├── interp/
│   ├── hooks.py              # Your clean activation patching framework
│   ├── logit_lens.py
│   ├── activation_patching.py
│   ├── steering.py
│   └── probes.py             # Stretch goal
├── notebooks/
│   ├── 01_logit_lens.ipynb
│   ├── 02_patching.ipynb
│   └── 03_steering.ipynb
├── figures/
│   ├── logit_lens.png
│   ├── patching_heatmap.png
│   └── steering.png
└── models/                   # .gitignored; HF links in README
```

**HuggingFace:** Upload your `sft-model` and `dpo-model` with model cards. This proves you shipped something.

---

## 6. Four-Week Execution Schedule

### Week 1: Foundation & SFT
- **Day 1–2:** Environment setup. Verify `mps` training works. Test inference speed on Qwen 0.5B. Fix any dtype issues (`bf16` is your friend).
- **Day 3–4:** Build and inspect the SFT dataset (1k examples). Write `build_sft_dataset.py`.
- **Day 5–6:** Run SFT. Debug. Evaluate on 10 held-out prompts to confirm instruction following.
- **Day 7:** Build the evaluation harness (`eval_refusal.py`). Test it on Base and SFT models.

### Week 2: Alignment & Validation
- **Day 8–9:** Curate the DPO dataset (400 refusal + 100 helpfulness pairs). This is manual work; write a clean filtering script.
- **Day 10–11:** Run DPO. Monitor the loss curves.
- **Day 12–13:** Run full evaluation. Compare SFT vs. DPO vs. Official Instruct on refusal rate and helpfulness. If DPO doesn't refuse well, debug (increase refusal pairs, tune β, or add a small SFT stage on refusal examples).
- **Day 14:** Buffer / documentation.

### Week 3: Mechanistic Analysis
- **Day 15–16:** Implement the hooking framework (`interp/hooks.py`). Write clean context managers for patching.
- **Day 17–18:** Run Logit Lens. Generate the plot. Write analysis in notebook.
- **Day 19–20:** Run Activation Patching (full residual, then attention-only, then MLP-only). Generate heatmap.
- **Day 21:** Run Steering Vector experiments. Generate the steering plot.

### Week 4: Synthesis & Publication
- **Day 22–23:** (Stretch) Run linear probing if you have time. Otherwise, deepen the analysis of Experiments 1–3.
- **Day 24–25:** Generate all final figures. Write the `README.md` as a mini-blog post: Motivation → Methods → Results → Takeaways.
- **Day 26–27:** Push models to HuggingFace. Ensure all scripts are reproducible (fix random seeds, document hyperparameters).
- **Day 28–29:** Polish. Add a "How to run on M4" section. Record a short demo or screenshot of results.
- **Day 30–31:** Buffer for unexpected bugs.

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **DPO model won't refuse** | Increase LoRA rank to 32. Ensure your SFT dataset has *zero* refusals. If still weak, add 50 explicit refusal examples to the SFT stage (but keep them minimal). |
| **MPS runs out of memory** | Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`. Use gradient checkpointing. Reduce max sequence length to 256 if needed. |
| **Mech interp shows "everything matters"** | This usually means your refusal signal is weak. If so, narrow your scope: instead of 20 prompts, find 5 *very strong* harmful prompts where refusal is near-deterministic. Or switch to studying a simpler behavior like "JSON formatting adherence" via DPO. |
| **Activation patching is too slow** | You only have 24 layers. Even on MPS, 24 forward passes × 50 prompts takes <30 minutes for a 0.5B model. |

---

## 8. How This Reads on Your Resume

> **Emergent Refusal Circuits in DPO-Aligned Sub-1B LLMs**  
> *Independent Research Project*
> - Built a complete post-training alignment pipeline (SFT → DPO) for a 500M-parameter model on Apple Silicon, inducing targeted refusal behaviors with no cloud compute.
> - Localized the causal refusal circuit to specific middle-layer attention/MLP components using activation patching and residual stream analysis.
> - Derived and validated steering vectors that could induce or suppress refusal on demand, demonstrating a mechanistic understanding of how preference learning rewrites model internals.
> - Open-sourced models, datasets, and reproducible training scripts on HuggingFace and GitHub.

---

## 9. Why This Project Is Strategically Strong

1. **It is not a tutorial.** You are not running `trl` and calling it a day. You are asking and answering a research question.
2. **It leverages your unique background.** Most RL/post-training candidates cannot do mech interp. Most interp people cannot train models. You are bridging the gap.
3. **It is timely.** DPO is the industry standard (replacing PPO in many places). Refusal is the central safety topic. Merging them is exactly what labs like Anthropic and DeepMind care about.
4. **It is honest about compute.** You are not claiming to train a 70B model. You are showing that deep insights can come from small models and careful analysis.

---

If you want, I can next help you write the exact `build_dpo_dataset.py` filtering logic, or the activation patching hook framework, or the training configs. Just tell me which piece you want to start with.

