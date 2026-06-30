# Act 4 — Multi-Variant Text Analysis and Generation (BERT, GPT, and GANs)

**Course:** Data Science 4
**Group 9:** Ira Louise Bayogos · Louis Patrick Jaso · Patrick Lawrence Molina

This repository contains the complete implementation, training history, and evaluation of three distinct NLP model variants — BERT (discriminative classification), GPT (generative language modeling), and a custom Text-GAN (adversarial generation) — trained and compared on a shared financial-news sentiment corpus.

---

## 1. Dataset

**Name:** Financial PhraseBank
**Citation:** Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology, 65*(4), 782–796. https://doi.org/10.1002/asi.23062

**Dataset repository link (Hugging Face):** https://huggingface.co/datasets/atrost/financial_phrasebank

**Description:** 4,846 English-language sentences extracted from financial news, labeled by 5–8 finance domain experts into three sentiment classes: Negative, Neutral, Positive. We use the pre-split `train` (3,100) / `validation` (776) / `test` (970) partitions provided by this Hugging Face mirror, since the original `financial_phrasebank` repository relies on a deprecated dataset loading script that is no longer supported by current versions of the `datasets` library.

The training split was further balanced via random oversampling (minority-class duplication with replacement) to correct for class imbalance prior to BERT fine-tuning; validation and test splits were left at their original, real-world class distribution. Full preprocessing code is in `notebooks/Act4_Group9_BayogosMolinaJaso.ipynb`, Part 0.

---

## 2. Repository Structure

```
.
├── README.md                                  <- this file
├── ANALYTICAL_DISCUSSION.md                   <- written analysis (tokenization + metric tradeoffs)
├── notebooks/
│   └── Act4_Group9_BayogosMolinaJaso.ipynb    <- main consolidated notebook (all 3 variants)
├── results/
   ├── bert/
   │   ├── bert_results.json                  <- final test metrics
   │   ├── bert_test_predictions.csv          <- per-sentence predictions
   │   └── confusion_matrix.png
   ├── gpt/
   │   ├── gpt_results.json                   <- baseline/final perplexity
   │   └── gpt_generations.csv                <- sample generations
   ├── gan/
   │   ├── gan_results.json                   <- BLEU scores, discriminator accuracy
   │   ├── gan_samples.csv                    <- generated synthetic sentences
   │   └── gan_training_history.csv           <- per-epoch G/D loss and accuracy
   └── comparison/
       ├── comparison_table.csv               <- cross-model metric summary
       ├── master_summary.json                <- consolidated results bundle
       ├── analysis.txt                       <- generated comparative analysis
       └── native_metrics_comparison.png

```

---

## 3. Model Variants Summary

| Variant | Base Architecture | Task | Training Paradigm |
|---|---|---|---|
| BERT | `bert-base-uncased` (Devlin et al., 2019) | 3-class sentiment classification | Supervised fine-tuning |
| GPT | `distilgpt2` (Sanh et al., 2019) | Causal language generation | Self-supervised fine-tuning |
| Text-GAN | Custom GRU Generator + CNN Discriminator | Adversarial text generation | Adversarial training from scratch |

## 4. Performance Evaluation Matrix

| Model Variant | Primary Metric (Precision / Recall / F1) | Generative Quality Metric (BLEU / ROUGE / Perplexity) | Training Time per Epoch | Key Observations / Constraints |
|---|---|---|---|---|
| **BERT** | P: 0.8300 · R: 0.8496 · F1: 0.8384 | N/A (Classification Task) | ~56.5 sec | Stable convergence; weakest on Positive recall (77.1%) due to overlap with Neutral language. |
| **GPT** | N/A (Generative Task) | Perplexity: 28.32 (from baseline 133.00) | ~40 sec | 4.70x perplexity reduction; mild overfitting after epoch 4; hallucinates specific entities. |
| **Text-GAN** | Discriminator Accuracy: 0.9999 | BLEU-2: 0.0900 · BLEU-4: 0.0023 | ~4–5 sec | Discriminator dominance from epoch 4; mode collapse; documented small-corpus GAN failure mode. |

Full per-class breakdowns, training curves, and qualitative samples are in the main notebook and `results/` folder.

---

## 5. How to Reproduce

1. Open `notebooks/Act4_Group9_BayogosMolinaJaso.ipynb` in Google Colab.
2. Set the runtime to T4 GPU (`Runtime → Change runtime type → T4 GPU`).
3. Run all cells top to bottom (`Runtime → Run all`).
4. All three model variants, their evaluations, and the cross-model comparison will execute in sequence within a single notebook.

**Dependencies:** `transformers`, `datasets`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `nltk`, `accelerate` — all installed automatically in the notebook's first cell.

---

## 6. Analytical Discussion

See [`ANALYTICAL_DISCUSSION.md`](./ANALYTICAL_DISCUSSION.md) for the full written analysis covering:
- How tokenization differences (WordPiece vs. BPE vs. custom word-level) affected each model's vocabulary coverage, training efficiency, and output quality.
- A detailed metric-tradeoff analysis explaining why GANs struggle with discrete text sequences relative to autoregressive models like GPT, grounded in both the theoretical literature and our own empirical results.

---

## 7. References

Full APA7 reference list is provided in the closing section of the main notebook and in `ANALYTICAL_DISCUSSION.md`.
