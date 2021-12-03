from transformers import pipeline

summarizer = pipeline("summarization", model="data/models/bart_attempt_3/checkpoint-60000")

ARTICLE = """ We present BART, a denoising autoencoder
for pretraining sequence-to-sequence models.
BART is trained by (1) corrupting text with an
arbitrary noising function, and (2) learning a
model to reconstruct the original text. It uses
a standard Tranformer-based neural machine
translation architecture which, despite its simplicity, can be seen as generalizing BERT (due
to the bidirectional encoder), GPT (with the
left-to-right decoder), and many other more recent pretraining schemes. We evaluate a number of noising approaches, finding the best performance by both randomly shuffling the order of the original sentences and using a novel
in-filling scheme, where spans of text are replaced with a single mask token. BART is
particularly effective when fine tuned for text
generation but also works well for comprehension tasks. It matches the performance of
RoBERTa with comparable training resources
on GLUE and SQuAD, achieves new stateof-the-art results on a range of abstractive dialogue, question answering, and summarization tasks, with gains of up to 6 ROUGE.
BART also provides a 1.1 BLEU increase over
a back-translation system for machine translation, with only target language pretraining. We
also report ablation experiments that replicate
other pretraining schemes within the BART
framework, to better measure which factors
most influence end-task performance.
"""
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))