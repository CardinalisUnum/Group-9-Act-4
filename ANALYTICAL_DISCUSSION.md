# Analytical Discussion — Act 4: Multi-Variant Text Analysis and Generation

**Group 9** — Ira Louise Bayogos, Louis Patrick Jaso, & Patrick Lawrence Molina
**Dataset:** Financial PhraseBank (Malo et al., 2014)


---

## 1. Tokenization Differences

| Model | Tokenizer | Vocab Size | Pretrained? | Key Constraint |
|---|---|---|---|---|
| **BERT** | WordPiece (Devlin et al., 2019) | ~30,000 | Yes | Splits unknown words into subwords — handles financial jargon gracefully despite general-domain pretraining |
| **GPT** | Byte-Pair Encoding (Radford et al., 2019) | ~50,000 | Yes | No native pad token — required manually setting `pad_token = eos_token`; used `max_length=64` to maximize training chunks from a small corpus |
| **Text-GAN** | Custom word-level (built from scratch) | 5,000 | No | Generator's output layer predicts over full vocab at every step — a larger vocab was not trainable from scratch on this corpus size; OOV words map to `<UNK>`, a blunter information loss than subword fallback |

**Takeaway:** BERT and GPT both inherit large, pretrained vocabularies that absorb unfamiliar financial terms gracefully. The GAN's smaller, lossier, from-scratch vocabulary is one of several compounding factors behind its weaker output quality.

---

## 2. Why GANs Struggle with Discrete Text (vs. GPT)

**The core problem:** text is discrete; gradient descent needs continuous functions. `argmax` token selection is not differentiable — it breaks the gradient path back to the Generator.

| | GPT | Text-GAN |
|---|---|---|
| **How it avoids/handles discreteness** | Teacher forcing — predicts a distribution, compares to ground truth via cross-entropy; never samples during training | Must sample during training — uses Gumbel-Softmax (Jang et al., 2017) as a differentiable approximation, with temperature annealed 1.0 → 0.5 |
| **Optimization structure** | Single network, single loss — smooth, predictable convergence | Two networks, no shared loss — adversarial minimax game, no convergence guarantee |
| **What we observed** | Stable loss curves across all 5 epochs | **Discriminator dominance** by epoch 4/30 — D-accuracy locked at 99.99%, G-loss climbed 0.71 → 7.69, resulting in mode collapse |

This is a well-documented GAN failure mode (Goodfellow, 2016; Salimans et al., 2016), not an implementation error.

### Why the evaluation metrics differ by model

| Model | Precision / Recall / F1 | BLEU / Perplexity |
|---|---|---|
| **BERT** | ✅ Real classification decision, clear ground truth | N/A — no generated sequence |
| **GPT** | N/A — no classification head | ✅ Perplexity: **133.00 → 28.32** (4.70× improvement) |
| **Text-GAN** | ✅ Discriminator accuracy *(diagnostic, not quality — ~50% would be the healthy outcome; our 99.99% signals collapse)* | ✅ BLEU-2: **0.0900** · BLEU-4: **0.0023** |

---

## 3. Key Results Snapshot

| Metric | BERT | GPT | Text-GAN |
|---|---|---|---|
| Headline score | Macro-F1: **0.8384** | Perplexity: **28.32** (from 133.00) | BLEU-2: **0.0900** |
| Secondary | Accuracy: 0.8515 | Improvement: 4.70× | D-accuracy: 0.9999 |
| Training behavior | Stable, mild overfit after ep. 2 | Stable, mild overfit after ep. 4 | Collapsed (D-dominance from ep. 4/30) |

Full figures (training curves, confusion matrix, cross-model comparison) are in [`ANALYTICAL_DISCUSSION.docx`](./ANALYTICAL_DISCUSSION.docx) and `results/`.

---

## References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019* (Vol. 1, pp. 4171–4186). ACL. https://doi.org/10.18653/v1/N19-1423

Goodfellow, I. (2016). NIPS 2016 tutorial: Generative adversarial networks. *arXiv*. https://arxiv.org/abs/1701.00160

Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization with Gumbel-Softmax. *ICLR*. https://arxiv.org/abs/1611.01144

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology, 65*(4), 782–796. https://doi.org/10.1002/asi.23062

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI*.

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. In *Advances in Neural Information Processing Systems* (Vol. 29, pp. 2234–2242). https://arxiv.org/abs/1606.03498
