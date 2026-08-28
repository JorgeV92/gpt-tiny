```text
Attention ✓
   ↓
KVCache          ← do this next
   ↓
FeedForward
   ↓
TransformerBlock
   ↓
Embedding + Transformer
   ↓
causal mask / prefill
   ↓
token-by-token generation
   ↓
load real LLaMA weights
   ↓
compare output with PyTorch GPT-Fast
   ↓
TinyJit
   ↓
benchmark PyTorch vs tinygrad
```