# Research Manual - Transfer Learning, Fine-Tuning, RAG Architectures, and NLP Fundamentals

Prepared by: Sameer Ahmed - AI Engineering Intern
Task: Intern Task 3 and 4 - Research Manual (Intermediate/Advanced Level)
Repository: https://github.com/SameerAhmedAI/Rag-Chatbot-LangGraph

---

## Table of Contents

1. Transfer Learning
2. Large Language Model (LLM) Fine-Tuning
3. Dataset Preparation for Fine-Tuning
4. LoRA (Low-Rank Adaptation)
5. QLoRA
6. Model Quantization
7. GPU Requirements for Training and Inference
8. GPU Memory Utilization and Optimization
9. Fine-Tuning vs RAG with Vector Databases
10. Naive RAG vs Graph RAG
11. Natural Language Processing (NLP) Concepts and Fundamentals
12. NLP Pipeline (Text Cleaning, Tokenization, Lemmatization, Stemming, Stop Words, Vectorization, Embeddings)
13. Word Embeddings (Word2Vec, GloVe, FastText)
14. Transformer Architecture
15. Attention Mechanism and Self-Attention
16. Large Language Models (LLMs)

---

## 1. Transfer Learning

### Concept

Transfer learning reuses a model already trained on a large, general dataset
and adapts it to a new, related task, instead of training a new model from
random weights. The intuition: a model trained on millions of images already
learned generic visual features (edges, textures, shapes) in its early
layers; those features transfer to a new image task, so only the later,
task-specific layers need retraining.

### How it works

Pretrained Model (trained on large dataset, e.g. ImageNet)
then Freeze early and generic layers (edges, textures, low-level patterns)
then Replace or unfreeze final layers (task-specific features)
then Fine-tune on smaller, task-specific dataset
then Adapted model - much less data and compute than training from scratch

### Practical example

In our own DL-RAG-Toolkit project, a ResNet18 pretrained on ImageNet
(1.2M images, 1000 classes) was adapted to CIFAR-10 (10 classes) using
only 2,800 training images. Early layers were frozen; only the final
residual block (layer4) and a new classifier head were retrained. Result:
84.1 percent test accuracy, achieved in 6 epochs - far faster convergence
and far less data than a CNN trained from scratch on the same task (which
took longer to train and scored lower relative to the effort involved).

### Advantages

- Dramatically reduces the amount of labeled data needed.
- Faster convergence (fewer epochs to reach strong accuracy).
- Lower compute cost than training from scratch.

### Limitations

- Works best when the source and target domains are related (ImageNet to
  natural images works well; ImageNet to medical X-rays works less well).
- Risk of overfitting when too many layers are unfrozen relative to the
  size of the new dataset (observed directly in our own ResNet18 result -
  75 percent of parameters were trainable against only 2,800 images,
  causing validation loss to plateau while training loss kept dropping).

### References

- He, Kaiming, et al. "Deep Residual Learning for Image Recognition." (2015)
- PyTorch transfer learning tutorial - pytorch.org/tutorials

---

## 2. Large Language Model (LLM) Fine-Tuning

### Concept

Fine-tuning takes a pretrained LLM (trained on general internet-scale text)
and further trains it on a smaller, task- or domain-specific dataset, so the
model's outputs shift toward the style, format, or knowledge needed for a
specific use case, e.g. customer support tone, legal document
summarization, or a company's internal terminology.

### How it works

Pretrained LLM (general language understanding)
then Task-specific labeled dataset (prompt to ideal response pairs)
then Supervised fine-tuning: adjust model weights via gradient descent
on the new dataset, typically at a low learning rate
then Optional alignment step: RLHF or DPO to further shape behavior
then Fine-tuned model - specialized, but weights are permanently changed

### Types of fine-tuning

- Full fine-tuning - every parameter in the model is updated. Most
  expressive, but requires the most compute and memory (must store
  gradients and optimizer states for every parameter).
- Parameter-efficient fine-tuning (PEFT) - only a small subset of
  parameters (or added parameters) are updated, e.g. LoRA (Section 4).
  Much cheaper, nearly comparable results for many tasks.

### Advantages

- Model internalizes domain knowledge and style permanently - no need to
  supply context at inference time.
- Can improve performance on narrow, well-defined tasks beyond what
  prompting alone achieves.

### Limitations

- Expensive and slow to update - any new information requires retraining.
- Risk of catastrophic forgetting (model loses general capabilities while
  specializing).
- Requires a labeled dataset, which is often the hardest part to produce.

### References

- Hugging Face fine-tuning guide - huggingface.co/docs/transformers/training
- OpenAI fine-tuning documentation - platform.openai.com/docs/guides/fine-tuning

---

## 3. Dataset Preparation for Fine-Tuning

### Concept

The quality of a fine-tuned model is bounded by the quality of its training
data - garbage in, garbage out applies more strongly here than almost
anywhere else in ML.

### Key steps

Raw data collection
then Cleaning (remove duplicates, fix formatting, filter low-quality examples)
then Structuring into prompt-response pairs (instruction-following format)
then Deduplication and diversity check
then Train, validation, test split (typically 80/10/10 or similar)
then Tokenization and format conversion (e.g. JSONL for most fine-tuning APIs)

### Practical considerations

- Volume: LoRA/PEFT fine-tuning can work with hundreds to low thousands
  of examples; full fine-tuning generally needs far more.
- Diversity: dataset should cover the range of inputs the model will see
  in production, not just easy examples.
- Label quality: noisy or inconsistent labels directly degrade fine-tuned
  model quality - this is usually the highest-leverage place to invest
  cleanup effort.
- Avoiding leakage: validation and test data must never overlap with
  training data, or evaluation metrics become meaningless.

### Advantages of careful preparation

- Directly improves fine-tuned model quality and consistency.
- Reduces the amount of training needed to reach target performance.

### Limitations

- Data preparation is often the most time-consuming part of a fine-tuning
  project - frequently more time than the actual training step.

### References

- Hugging Face datasets documentation - huggingface.co/docs/datasets

---

## 4. LoRA (Low-Rank Adaptation)

### Concept

LoRA is a parameter-efficient fine-tuning technique. Instead of updating
every weight in a large pretrained model, LoRA freezes the original
weights entirely and injects small, trainable adapter matrices into each
layer. These adapters are low-rank (much smaller than the original weight
matrix), so they require far fewer trainable parameters, often less than
1 percent of the full model.

### How it works

Original weight matrix W (frozen, unchanged)
plus Low-rank update: delta_W = A x B (A: d x r matrix, B: r x d matrix, r much less than d)
equals Effective weight during inference: W + delta_W

Only A and B are trained; W stays frozen throughout.

The rank r (e.g. 4, 8, 16) controls the tradeoff between adapter
expressiveness and parameter count - higher rank means more trainable
parameters and more capacity to adapt, lower rank means faster and
cheaper training with less capacity.

### Advantages

- Drastically reduces trainable parameter count and memory needed for
  gradients and optimizer states.
- Multiple LoRA adapters can be trained for different tasks and swapped
  in and out without duplicating the full base model.
- Comparable performance to full fine-tuning on many tasks, at a fraction
  of the cost.

### Limitations

- May underperform full fine-tuning on tasks requiring deep, broad shifts
  in model behavior.
- Choosing rank and which layers to adapt requires some experimentation.

### References

- Hu, Edward, et al. "LoRA: Low-Rank Adaptation of Large Language Models." (2021)
- Hugging Face PEFT library - huggingface.co/docs/peft

---

## 5. QLoRA

### Concept

QLoRA combines LoRA with quantization (Section 6): the frozen base model is
loaded in a heavily compressed, low-precision format (typically 4-bit)
to drastically reduce memory footprint, while LoRA adapters are still
trained in higher precision on top of it. This makes it possible to
fine-tune very large models (e.g. 65B+ parameters) on a single consumer or
prosumer GPU, which would otherwise require multiple high-end GPUs.

### How it works

Base model weights quantized to 4-bit (e.g. NF4 format) and frozen
plus LoRA adapter matrices (A, B) kept in higher precision (e.g. bfloat16)
then Forward and backward pass: 4-bit weights dequantized on-the-fly for
computation, gradients only flow into the LoRA adapters
then Result: full-model-scale fine-tuning at a fraction of the memory cost

### Advantages

- Enables fine-tuning of very large models on hardware that couldn't
  otherwise fit them in memory at all.
- Maintains near full-fine-tuning-quality results in many published
  benchmarks, despite the aggressive compression.

### Limitations

- Slightly slower per-step than standard LoRA due to the quantize and
  dequantize overhead during computation.
- Requires careful implementation (e.g. double quantization, paged
  optimizers) to actually realize the memory savings in practice, not
  a drop-in replacement without the right tooling (e.g. bitsandbytes).

### References

- Dettmers, Tim, et al. "QLoRA: Efficient Finetuning of Quantized LLMs." (2023)
- bitsandbytes library - github.com/TimDettmers/bitsandbytes

---

## 6. Model Quantization

### Concept

Quantization reduces the numerical precision used to store and compute a
model's weights (and sometimes activations), for example converting
32-bit floating point weights (FP32) down to 8-bit integers (INT8) or
4-bit formats. This shrinks model size and speeds up inference, at the
cost of some numerical precision.

### Common precision levels

FP32 (32 bits) - Default training precision, highest accuracy
FP16 / BF16 (16 bits) - Common for training/inference speedup, minimal accuracy loss
INT8 (8 bits) - Common for inference-time compression
INT4 / NF4 (4 bits) - Aggressive compression, used in QLoRA and local LLM deployment

### How it works (simplified)

Full-precision weight (e.g. 32-bit float)
then Determine scale and zero-point mapping for the target bit-width
then Map continuous weight values to a smaller discrete set of values
then Quantized weight (e.g. 4-bit integer plus scale factor)
then At inference: dequantize on-the-fly (or compute directly in low precision)

### Advantages

- Significantly reduces model file size and memory footprint.
- Speeds up inference, especially on hardware with optimized low-precision
  compute paths.
- Enables running large models on consumer-grade hardware.

### Limitations

- Some accuracy loss is possible, particularly at very low bit-widths
  (e.g. 4-bit) without careful calibration.
- Not all hardware and software stacks support every quantization format
  equally well.

### References

- Dettmers, Tim, et al. "LLM.int8(): 8-bit Matrix Multiplication for
  Transformers at Scale." (2022)
- Hugging Face quantization guide - huggingface.co/docs/transformers/quantization

---

## 7. GPU Requirements for Training and Inference

### Concept

GPU requirements differ significantly between training (or fine-tuning) a
model and simply running inference on an already-trained model - training
requires far more memory because it must store gradients and optimizer
states in addition to the model weights themselves.

### Rough memory breakdown (full fine-tuning, FP32)

Total GPU memory needed approximately equals:
Model weights (params x 4 bytes)
plus Gradients (params x 4 bytes)
plus Optimizer states (Adam: params x 8 bytes, for 2 moment estimates)
plus Activations (depends on batch size and sequence length)

For a 7B parameter model in full FP32 fine-tuning, this rough formula
already exceeds 100GB - well beyond a single consumer GPU - which is
exactly the practical motivation for PEFT methods like LoRA and QLoRA
(Sections 4-5), which drastically cut the gradient and optimizer-state
portion of that total.

### Inference-only requirements

Inference needs only the model weights (plus a small amount of memory for
activations during the forward pass) - no gradients or optimizer states.
A 7B parameter model in FP16 needs roughly 14GB just for weights; in 4-bit
quantized form, roughly 4GB - the difference that makes local LLM
deployment on consumer GPUs practical.

### Advantages of understanding this tradeoff

- Directly informs hardware budgeting decisions before a project starts.
- Explains why quantization and PEFT methods exist as practical
  necessities, not just optimizations.

### Limitations

- Real-world memory usage varies with framework overhead, batch size,
  sequence length, and specific optimizer choice - these are estimates,
  not exact figures.

### References

- Hugging Face model memory estimation guide -
  huggingface.co/docs/transformers/perf_train_gpu_one

---

## 8. GPU Memory Utilization and Optimization

### Concept

Beyond simply having enough GPU memory, several techniques exist to use
available memory more efficiently, enabling larger models or batch sizes
on the same hardware.

### Key techniques

- Gradient checkpointing - instead of storing every intermediate
  activation for the backward pass, recompute some of them on-the-fly
  during backpropagation. Trades compute time for memory savings.
- Mixed precision training - use FP16/BF16 for most computation while
  keeping a FP32 master copy of weights for numerical stability, cutting
  activation and gradient memory roughly in half versus pure FP32.
- Gradient accumulation - simulate a larger effective batch size by
  accumulating gradients over several smaller batches before updating
  weights, avoiding the memory cost of a single large batch.
- Paged optimizers (used in QLoRA) - offload optimizer states to CPU
  memory when not actively needed, paging them back to GPU on demand.
- Model and tensor parallelism - split a model's layers or individual
  tensors across multiple GPUs when a model doesn't fit on one device.

### Practical example from our own project

On our i7-11800H laptop with a T1200 4GB GPU, none of the intermediate
deep learning models (CNN, Transfer Learning) could realistically be
trained on GPU at full batch size - this directly motivated CPU-only
training with reduced dataset subsets and smaller epoch counts, a real
demonstration of working within actual hardware constraints rather than
theoretical ones.

### Advantages

- Enables training and fine-tuning larger models than raw memory would
  otherwise allow.
- Often has minimal or no accuracy cost (e.g. mixed precision, gradient
  checkpointing) in exchange for the memory savings.

### Limitations

- Some techniques trade memory for speed (e.g. gradient checkpointing
  slows down training by requiring recomputation).
- Adds implementation complexity versus a naive training loop.

### References

- PyTorch gradient checkpointing docs - pytorch.org/docs/stable/checkpoint.html
- Hugging Face performance guide - huggingface.co/docs/transformers/perf_train_gpu_one

---

## 9. Fine-Tuning vs RAG with Vector Databases

### Concept comparison

Dimension: How knowledge is added
- Fine-Tuning: Baked into model weights via training
- RAG: Retrieved at query time from an external store

Dimension: Update speed
- Fine-Tuning: Slow, requires retraining
- RAG: Fast, just re-index new documents

Dimension: Cost to update
- Fine-Tuning: High (compute plus data prep)
- RAG: Low (embed plus index new content)

Dimension: Source transparency
- Fine-Tuning: Opaque, can't easily cite what was learned
- RAG: Transparent, retrieved chunks can be cited directly

Dimension: Hallucination risk
- Fine-Tuning: Can still hallucinate on out-of-distribution queries
- RAG: Can be constrained to only answer from retrieved context

Dimension: Best suited for
- Fine-Tuning: Style, tone, format, narrow specialized behavior
- RAG: Fast-changing or large factual knowledge bases

Dimension: Data requirements
- Fine-Tuning: Needs labeled prompt-response pairs
- RAG: Needs raw documents, no labeling required

### How this project (RAG Chatbot with LangGraph) applies this

This project deliberately chose RAG over fine-tuning specifically because
the task requires answering questions grounded in arbitrary, user-uploaded
documents that change per session - fine-tuning a model per uploaded
document set would be prohibitively slow and expensive. RAG allows
instant grounding in new content the moment it is uploaded and embedded.

### When to combine both

In practice, the two approaches are complementary rather than mutually
exclusive: fine-tuning can shape a model's tone, output format, or
domain-specific reasoning style, while RAG supplies the specific,
up-to-date factual content the model reasons over. A production system
might fine-tune a model to always cite sources in a specific format, then
use RAG to supply the sources themselves.

### References

- Lewis, Patrick, et al. "Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks." (2020)
- Pinecone RAG vs fine-tuning guide - pinecone.io/learn

---

## 10. Naive RAG vs Graph RAG

### Concept comparison

Naive RAG (the architecture used in this project): documents are
chunked, embedded, and stored in a vector database. At query time, the
top-k most semantically similar chunks are retrieved via similarity
search and passed to the LLM as context.

Graph RAG: documents (or extracted entities and relationships within
them) are represented as a knowledge graph - nodes for entities, edges
for relationships between them. Retrieval can then traverse relationships
(multi-hop reasoning), not just find semantically similar isolated chunks.

### Architecture comparison

Naive RAG flow:
Document then Chunk then Embed then Vector DB
Query then Embed then Similarity Search (top-k chunks) then LLM

Graph RAG flow:
Document then Entity/Relationship Extraction then Knowledge Graph
Query then Identify relevant entities then Traverse graph relationships
then Gather multi-hop context then LLM

### Comparison table

Dimension: Retrieval unit
- Naive RAG: Isolated text chunks
- Graph RAG: Connected entities and relationships

Dimension: Multi-hop reasoning
- Naive RAG: Weak, each chunk retrieved independently
- Graph RAG: Strong, can traverse relationships across documents

Dimension: Setup complexity
- Naive RAG: Low, chunk, embed, store
- Graph RAG: High, requires entity and relationship extraction pipeline

Dimension: Best suited for
- Naive RAG: Direct factual lookup within a document
- Graph RAG: Questions requiring reasoning across multiple connected facts

Dimension: Example weakness
- Naive RAG: "How does X relate to Y across three different documents?"
  often retrieves each mention separately, missing the connection
- Graph RAG: Naturally surfaces the connecting path between X and Y

### Relevance to this project

This project uses Naive RAG throughout, which is well-suited to its
actual use case - answering direct questions grounded in a single
uploaded document (or a small number of documents) - but has a known
limitation, documented directly in this project's own testing (see
README.md, Section 9, "Challenges and Solutions"): retrieval is limited
to whatever chunks are most semantically similar to the query, and does
not reason across explicit relationships between entities mentioned in
different parts of a document or across documents. Graph RAG is listed as
a Future Improvement for exactly this reason.

### Advantages of Naive RAG

- Simple to implement and reason about.
- Fast to set up, no entity extraction pipeline needed.
- Sufficient for most direct question-answering use cases.

### Limitations of Naive RAG

- Struggles with questions requiring connecting information across
  multiple, non-adjacent chunks or documents.
- No structural understanding of how entities relate to each other.

### Advantages of Graph RAG

- Enables genuine multi-hop reasoning across connected facts.
- Can answer "how are X and Y related" style questions more reliably.

### Limitations of Graph RAG

- Significantly more complex to build and maintain (entity extraction,
  relationship extraction, graph construction, graph traversal logic).
- Extraction quality directly bounds retrieval quality - errors in
  entity or relationship extraction propagate into retrieval mistakes.

### References

- Microsoft Research, "GraphRAG: A new approach for discovery using
  complex information." (2024) - microsoft.com/research
- Neo4j GraphRAG documentation - neo4j.com/developer/graph-data-science

---

## 11. Natural Language Processing (NLP) Concepts and Fundamentals

### Concept

NLP is the field concerned with enabling computers to process, understand,
and generate human language. Unlike structured data (numbers, categories),
text is unstructured, ambiguous, and context-dependent - the same word can
mean different things depending on surrounding words, and meaning often
depends on world knowledge the model was never explicitly given. NLP sits
at the intersection of linguistics, statistics, and machine learning, and
underlies every component of a RAG system: understanding the user's
question, splitting documents into meaningful chunks, and representing text
numerically so it can be compared for similarity.

### How it works

Raw text (unstructured, ambiguous)
then Preprocessing (cleaning, tokenization - see Section 12)
then Numerical representation (vectorization/embeddings - see Section 12/13)
then Model processing (classification, generation, retrieval, etc.)
then Structured output (answer, label, ranked results)

### Practical example

This entire project is an applied NLP system end to end: user questions are
tokenized and embedded (all-MiniLM-L6-v2), compared against embedded document
chunks via cosine similarity in ChromaDB, and the retrieved text is passed to
a transformer-based LLM (GPT-OSS-120B via Groq) for answer generation. Every
stage - ingestion, chunking, embedding, retrieval, generation - is a distinct
NLP sub-task chained together.

### Advantages

- A mature field with well-established techniques and pretrained models
  available for nearly every sub-task (tokenization, embeddings, generation).
- Modern transformer-based approaches generalize well across domains without
  needing hand-crafted linguistic rules.

### Limitations

- Language is genuinely ambiguous (sarcasm, idioms, context-dependent
  meaning) - no NLP system achieves perfect understanding.
- Performance is strongly tied to the quality and domain-match of training
  data; a model trained on general web text can underperform on specialized
  technical or legal language without adaptation.

### References

- Jurafsky, Daniel and Martin, James H. "Speech and Language Processing."
  (3rd edition draft) - web.stanford.edu/~jurafsky/slp3
- spaCy documentation - spacy.io/usage/linguistic-features

---

## 12. NLP Pipeline (Text Cleaning, Tokenization, Lemmatization, Stemming, Stop Words, Vectorization, Embeddings)

### Concept

A typical NLP pipeline is a sequence of preprocessing steps that convert raw
text into a numerical form a model can consume. Not every application uses
every step - modern transformer-based systems (including this project) skip
several classical steps entirely, since subword tokenization and learned
embeddings handle much of what stemming/lemmatization/stop-word removal used
to do by hand.

### Pipeline stages

- Text cleaning: removing noise - HTML tags, extra whitespace, special
  characters - so the model sees clean, consistent text.
- Tokenization: splitting text into units (words, subwords, or
  characters) the model can process. Modern LLMs use subword tokenization
  (e.g. Byte-Pair Encoding), which handles rare/unseen words by breaking
  them into known sub-pieces rather than failing outright.
- Stop word removal: discarding very common, low-information words
  ("the", "is", "a"). Common in classical NLP (search engines, bag-of-words
  models); generally skipped in transformer pipelines, since the model can
  learn to weight these words appropriately itself.
- Stemming: crudely truncating words to a root form (e.g. "running" to
  "run") using fixed rules - fast but can produce non-words (e.g. "studies"
  to "studi").
- Lemmatization: reducing words to their true dictionary root ("better" to
  "good") using vocabulary and grammar rules - slower than stemming but
  linguistically accurate.
- Vectorization: converting text into numbers - classically via
  bag-of-words or TF-IDF (frequency-based, no notion of meaning).
- Embeddings: converting text into dense numerical vectors that capture
  semantic meaning, such that similar meanings produce similar vectors -
  the modern replacement for simple vectorization. See Section 13.

### Practical example from our own project

The ingestion pipeline in this project (LoaderFactory to
RecursiveCharacterTextSplitter) performs cleaning (via each format-specific
loader) and chunking, but deliberately skips stemming, lemmatization, and
stop-word removal - these are unnecessary and can even hurt retrieval
quality with modern sentence-transformer embeddings, since the embedding
model was trained on natural, unstemmed text and expects the same at
inference time. Tokenization happens implicitly inside both the embedding
model (all-MiniLM-L6-v2) and the LLM (GPT-OSS-120B), not as a manual
pipeline step in this codebase.

### Advantages

- A well-defined pipeline makes text preprocessing systematic and
  reproducible across a large corpus.
- Classical steps (stemming, stop-word removal) are cheap and still useful
  for lightweight, non-neural applications (e.g. keyword search).

### Limitations

- Stemming can produce incorrect roots and merge unrelated words
  (over-stemming), hurting downstream accuracy.
- Applying classical preprocessing (stop-word removal, stemming) before
  a transformer-based embedding model can remove information the model
  relies on, since these models were trained on natural text - a common
  mistake when combining old and new NLP techniques.

### References

- Manning, Christopher D., et al. "Introduction to Information Retrieval."
  Cambridge University Press (2008) - nlp.stanford.edu/IR-book
- Hugging Face Tokenizers documentation - huggingface.co/docs/tokenizers

---

## 13. Word Embeddings (Word2Vec, GloVe, FastText)

### Concept

Word embeddings represent words as dense numerical vectors (typically
100-300 dimensions) such that words with similar meanings are positioned
close together in vector space. This replaced older sparse representations
(one-hot encoding, bag-of-words), which treated every word as equally
unrelated to every other word and produced enormous, mostly-empty vectors.

### The three classical approaches

- Word2Vec (Mikolov et al., 2013): learns embeddings by predicting a
  word from its surrounding context (CBOW) or predicting context from a
  word (Skip-gram). Famous for capturing analogies algebraically
  (king - man + woman is approximately equal to queen).
- GloVe (Global Vectors, Pennington et al., 2014): learns embeddings
  from global word co-occurrence statistics across the entire corpus,
  rather than local context windows - combines the strengths of matrix
  factorization and local context methods.
- FastText (Bojanowski et al., 2017, Facebook AI): extends Word2Vec by
  representing each word as a bag of character n-grams, so it can generate
  reasonable embeddings for misspelled or out-of-vocabulary words by
  composing them from known sub-word pieces.

### How this relates to modern embeddings

Word2Vec/GloVe/FastText produce one fixed vector per word regardless of
context ("bank" gets the same vector in "river bank" and "bank account").
Modern transformer-based embeddings (used in this project - see Section 11)
produce contextual embeddings, where the same word gets different vectors
depending on surrounding text. This is the key advance that made classical
word embeddings largely obsolete for high-accuracy retrieval tasks.

### Advantages

- Dramatically more compact and semantically meaningful than one-hot or
  bag-of-words representations.
- Fast to train and use; still useful for lightweight applications where
  full transformer models are too costly.
- FastText's subword approach handles rare and misspelled words gracefully.

### Limitations

- Static, context-independent: cannot distinguish different meanings of
  the same word based on context, a real limitation for ambiguous language.
- Largely superseded by contextual embeddings (BERT-style, sentence
  transformers) for tasks like semantic search and RAG retrieval, where
  this project uses all-MiniLM-L6-v2 rather than Word2Vec/GloVe/FastText.

### References

- Mikolov, Tomas, et al. "Efficient Estimation of Word Representations in
  Vector Space." (2013)
- Pennington, Jeffrey, et al. "GloVe: Global Vectors for Word
  Representation." (2014) - nlp.stanford.edu/projects/glove
- Bojanowski, Piotr, et al. "Enriching Word Vectors with Subword
  Information." (2017)

---

## 14. Transformer Architecture

### Concept

The Transformer (Vaswani et al., 2017, "Attention Is All You Need")
replaced recurrent architectures (RNN, LSTM - see the DL-RAG-Toolkit
technical report referenced elsewhere in this repository) as the dominant
architecture for sequence modeling. Its key insight: instead of processing
a sequence one token at a time (as RNNs/LSTMs do, which is slow and
struggles with long-range dependencies), a Transformer processes an entire
sequence in parallel, using attention (see Section 15) to let every token
directly relate to every other token regardless of distance.

### Architecture (encoder-decoder)

Input Embeddings + Positional Encoding
then Encoder stack (Multi-Head Self-Attention, then Feed-Forward Network,
repeated N times, each with residual connections and layer normalization)
then Decoder stack (Masked Self-Attention, then Encoder-Decoder Attention,
then Feed-Forward Network, repeated N times)
then Output projection (linear layer plus softmax over vocabulary)

Modern LLMs (including the GPT-OSS-120B model used in this project) are
decoder-only Transformers - they drop the encoder and use only the masked
self-attention decoder stack, since generation (predicting the next token)
does not require a separate encoding pass the way translation originally did.

### Why this matters for RAG systems

Both halves of a RAG pipeline lean on Transformers: the embedding model
(all-MiniLM-L6-v2, an encoder-only Transformer) converts text into vectors
for retrieval, and the generation model (GPT-OSS-120B, a decoder-only
Transformer) produces the final answer from retrieved context.

### Advantages

- Fully parallelizable during training (unlike RNNs, which must process
  tokens sequentially), enabling much larger models trained on much more data.
- Handles long-range dependencies far better than RNN/LSTM architectures,
  since attention connects any two tokens directly regardless of distance.

### Limitations

- Self-attention has quadratic computational cost in sequence length
  (doubling context length roughly quadruples attention compute), which is
  why context window size is a meaningful cost and latency constraint.
- Requires large amounts of training data and compute to reach strong
  performance from scratch, though this project only performs inference
  against a pretrained model rather than training a Transformer directly.

### References

- Vaswani, Ashish, et al. "Attention Is All You Need." (2017)
- Jay Alammar, "The Illustrated Transformer" - jalammar.github.io/illustrated-transformer

---

## 15. Attention Mechanism and Self-Attention

### Concept

Attention lets a model dynamically weigh how much each part of the input
matters when producing a given output, rather than treating all input
positions equally. Self-attention is attention applied within a single
sequence: for each token, the mechanism computes how strongly it should
attend to every other token in the same sequence, including itself.

### How it works (simplified)

Each token's embedding is projected into three vectors: Query (Q), Key (K),
and Value (V).

For each token: compute similarity (dot product) between its Query and
every token's Key
then Scale and apply softmax to get attention weights (summing to 1)
then Compute a weighted sum of all Value vectors using those weights
then Result: a new representation of the token, informed by relevant
context from the rest of the sequence

Multi-head attention repeats this process several times in parallel with
different learned projections, letting the model capture different types
of relationships simultaneously (e.g. one head might track syntactic
structure while another tracks coreference).

### Practical relevance

Self-attention is why a Transformer-based embedding model can produce a
genuinely different vector for the word "bank" depending on whether nearby
tokens relate to rivers or finance - each token's representation is built
by attending to its actual context, not looked up from a fixed table
(contrast this with Word2Vec/GloVe in Section 13). This same mechanism, in
the LLM used for generation, is what lets the model correctly connect a
pronoun like "that" in a follow-up question to the specific entity it
refers to earlier in a long context window.

### Advantages

- Directly models relationships between any two positions in a sequence,
  regardless of distance, addressing the core weakness of RNN/LSTM
  architectures.
- Interpretable to a degree - attention weights can be visualized to see
  which tokens a model focused on for a given prediction.

### Limitations

- Computationally expensive at scale (quadratic in sequence length, as
  noted in Section 14).
- High attention weight does not always equal causal importance -
  attention visualizations can be suggestive but are not a rigorous
  explanation of model behavior.

### References

- Vaswani, Ashish, et al. "Attention Is All You Need." (2017)
- Bahdanau, Dzmitry, et al. "Neural Machine Translation by Jointly Learning
  to Align and Translate." (2014) - the earlier attention mechanism that
  inspired the Transformer's self-attention

---

## 16. Large Language Models (LLMs)

### Concept

LLMs are large, decoder-only Transformer models (see Section 14) trained on
massive text corpora to predict the next token in a sequence. At sufficient
scale, this simple training objective produces models capable of a wide
range of downstream tasks - question answering, summarization, reasoning,
code generation - without task-specific training, through prompting alone.

### How this project uses an LLM

This project uses GPT-OSS-120B (via the Groq API) for two distinct roles:

- Generation: producing the final grounded answer from retrieved document
  context (the core RAG task).
- Auxiliary reasoning steps: the query rewriter (resolving follow-up
  pronouns using history) and the LangGraph router and critique nodes
  (classifying questions and self-checking answers) all call the same LLM
  with different, narrowly-scoped prompts - demonstrating that a single LLM
  can serve multiple distinct roles within one application through prompt
  engineering alone, without separate fine-tuned models for each task.

Note: this project originally used Groq's llama-3.3-70b-versatile,
which Groq decommissioned on August 16, 2026. The project was migrated to
openai/gpt-oss-120b, Groq's recommended replacement - see the main
README's Challenges and Solutions section for the full account of this
migration.

### Advantages

- A single pretrained LLM can perform many tasks via prompting, avoiding
  the cost of training or fine-tuning separate models per task.
- Strong few-shot and zero-shot performance on tasks the model was never
  explicitly trained for.

### Limitations

- Hallucination: LLMs can generate fluent, confident-sounding text that is
  factually incorrect or entirely fabricated, especially when given
  insufficient or no grounding context - a failure mode directly
  encountered and fixed during this project's own testing (see the main
  README, Challenge 5).
- No inherent access to information outside training data or the provided
  context window - this is precisely the gap RAG is designed to close (see
  Section 9, Fine-Tuning vs RAG with Vector Databases).
- Inference cost and latency scale with model size and context length,
  directly motivating the quantization and GPU optimization techniques
  covered in Sections 6-8 of this manual.

### References

- Brown, Tom, et al. "Language Models are Few-Shot Learners" (GPT-3 paper).
  (2020)
- Groq documentation - console.groq.com/docs
- OpenAI, "gpt-oss Model Card" - openai.com/index/gpt-oss

---

## Summary

This manual covers the foundational techniques for adapting and deploying
large models (transfer learning, fine-tuning, LoRA, QLoRA, quantization)
alongside the hardware considerations that make these techniques
practical (GPU requirements and memory optimization), and includes a
direct comparison of the two dominant strategies for grounding an LLM in
external knowledge - fine-tuning versus RAG - and the two dominant RAG
architectures - Naive RAG versus Graph RAG. It closes with the NLP
foundations underlying all of the above: the classical and modern NLP
pipeline, word embeddings, the Transformer architecture, the attention
mechanism that makes Transformers work, and large language models
themselves - tying each concept back to a concrete role it plays in this
project (embedding-based retrieval, LLM-based generation, and the
multi-role prompting used by the query rewriter, router, and critique
nodes). The RAG Chatbot with LangGraph project (this repository) is a
direct, working application of nearly every concept in this manual - the
Naive RAG approach described in Section 10, the Transformer-based
embedding and generation models described in Sections 14-16, and the
real limitations of both, documented honestly based on actual testing
rather than assumed theoretically.