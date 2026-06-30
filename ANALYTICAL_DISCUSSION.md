# Analytical Discussion — Act 4: Multi-Variant Text Analysis and Generation

**Group 9** — Ira Louise Bayogos, Louis Patrick Jaso, & Patrick Lawrence Molina
**Dataset:** Financial PhraseBank (Malo et al., 2014)

---

## 1. Tokenization Differences Across the Three Models

Even though BERT, GPT, and the Text-GAN were trained on the same corpus, each one tokenized that corpus in a completely different way, and this had real consequences on what each model could and couldn't learn.

BERT uses WordPiece, a subword tokenizer with roughly 30,000 tokens (Devlin et al., 2019). When it runs into a word it doesn't recognize as a whole unit — financial jargon, for instance — it breaks the word into smaller pieces it does recognize instead of just discarding it. That matters here because BERT's original pretraining (Wikipedia and BooksCorpus) was never specifically about finance, yet it still handled our domain reasonably well, and the subword fallback is a big reason why.

GPT uses Byte-Pair Encoding instead, with a much larger vocabulary of about 50,000 tokens (Radford et al., 2019). One annoying difference we ran into early: GPT-2 has no built-in padding token, so we had to manually set `tokenizer.pad_token = tokenizer.eos_token` before anything would run — a fix that's specific to GPT-style tokenizers and has no real equivalent on the BERT side. We also tokenized GPT's input at a shorter max length (64 tokens instead of BERT's 128), mainly because our corpus is small (under 5,000 sentences) and shorter chunks meant more training examples out of the same data.

The Text-GAN's tokenizer is the odd one out — we built it ourselves, word by word, capped at 5,000 vocabulary entries. This wasn't a stylistic choice; the Generator's output layer has to predict a probability over the entire vocabulary at every position, and asking it to learn a 30K or 50K-wide distribution from scratch, with no pretrained weights to lean on, simply wasn't realistic given our corpus size. The tradeoff is that anything outside the top 5,000 words gets mapped to `<UNK>` and effectively lost, which is a much blunter form of information loss than what BERT or GPT experience. This is part of why the GAN's generated samples end up so narrow and repetitive — the model never had access to the same vocabulary richness the other two did.

In short: BERT and GPT both got to borrow large, pretrained vocabularies that could gracefully absorb unfamiliar financial terms. The GAN had no such safety net, and its smaller, lossier, custom vocabulary is one of several compounding factors behind its weaker output quality.

---

## 2. Why GANs Struggle with Discrete Text (and GPT Doesn't)

The short version of why this is hard: text is made of discrete tokens, but gradient descent needs continuous, differentiable functions to work. A GAN's Generator eventually has to pick one specific word at each position, and the obvious way to do that — `argmax` over a probability distribution — isn't differentiable. Once you apply `argmax`, the gradient connection back to the Generator's weights is broken, and the Discriminator's feedback has nowhere to go.

GPT sidesteps this problem entirely during training. Because it's trained with teacher forcing, it never actually has to sample a token during the training loop — it just predicts a probability distribution and compares it against the known correct next word using ordinary cross-entropy loss. Sampling only happens later, at generation time, by which point training is already done. There's no discreteness problem to solve in the first place.

Our GAN couldn't avoid the problem, so we used Gumbel-Softmax (Jang et al., 2017) — a way of approximating discrete sampling with a continuous, differentiable function, controlled by a temperature parameter we slowly annealed from 1.0 down to 0.5 over training. It works, but it's an approximation, and it adds its own instability on top of an already harder optimization problem.

That harder problem is the second piece of the puzzle. GPT's fine-tuning is a single network minimizing a single, well-behaved loss function — which is exactly why our GPT training curves looked clean and predictable. A GAN, on the other hand, is two networks pulling in opposite directions with no shared loss to track convergence against. We saw this play out directly in our own results: by epoch 4 of 30, our Discriminator's accuracy had locked at 99.99%, meaning it was correctly identifying almost every fake sample the Generator produced. Once that happens, the Discriminator stops giving the Generator any useful signal — every fake gets roughly the same "this is obviously fake" verdict regardless of what the Generator does differently — and the Generator has no real direction to improve in. Our Generator's loss climbed from 0.71 at the start to 7.69 by the end, which is the loss curve equivalent of a student getting a failing grade no matter how they answer the question.

This is a well-known issue in the GAN literature, usually called Discriminator dominance (Goodfellow, 2016; Salimans et al., 2016), and our results line up with it closely — including the mode collapse we saw in the generated text, where the same handful of low-information tokens (most noticeably the fragment "hel," likely a leftover piece of "Helsinki" from this Finnish-sourced corpus) kept reappearing across nearly every sample.

This difference also explains why the assignment's evaluation table asks for different metrics per model. Precision, Recall, and F1 make sense for BERT because it's making a real classification decision with clear right answers. They also make sense for the GAN's Discriminator, but only as a binary real-vs-fake check — and importantly, a Discriminator accuracy near 50% would actually be the *good* outcome here, since it means the Generator is fooling it about half the time. Our 99.99% is the opposite signal: a Discriminator that's stopped being useful as a training partner. GPT has neither a classification head nor a Discriminator, so Precision/Recall/F1 don't apply to it at all — which is why the assignment table correctly marks that cell N/A, and why we relied on Perplexity (133.00 baseline, dropping to 28.32 after fine-tuning) as its main quality signal instead, alongside the BLEU-2 (0.0900) and BLEU-4 (0.0023) scores we computed for the GAN's generated text against the real corpus.

Put simply, GPT had an easier path because its training objective was built to avoid the discreteness problem altogether, while our GAN had to confront it head-on with an imperfect workaround, on top of a fundamentally less stable two-network training setup. The Discriminator-dominance collapse we observed isn't really a coding mistake — it's a known, documented behavior of adversarial text generation on small datasets, and our results are a fairly textbook example of it.

---

## References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (Vol. 1, pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1423

Goodfellow, I. (2016). NIPS 2016 tutorial: Generative adversarial networks. *arXiv*. https://arxiv.org/abs/1701.00160

Jang, E., Gu, S., & Poole, B. (2017). Categorical reparameterization with Gumbel-Softmax. *International Conference on Learning Representations*. https://arxiv.org/abs/1611.01144

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology, 65*(4), 782–796. https://doi.org/10.1002/asi.23062

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI*. https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., & Chen, X. (2016). Improved techniques for training GANs. In D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, & R. Garnett (Eds.), *Advances in Neural Information Processing Systems* (Vol. 29, pp. 2234–2242). Curran Associates. https://arxiv.org/abs/1606.03498
