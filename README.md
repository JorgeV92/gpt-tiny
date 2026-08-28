# gpt-tiny 

A fast gpt inference model built with `tinygrad`

## Tokenization

`tokenizer.py` includes a Llama 3 tokenizer.

```python
from tokenizer import Llama3Tokenizer, encode_tensor

tokenizer = Llama3Tokenizer.from_file(
    "checkpoints/meta-llama/Meta-Llama-3-8B/tokenizer.model"
)
prompt = encode_tensor(tokenizer, "Hello from tinygrad", add_bos=True)
```

Chat prompts can be encoded without coupling tokenization to model execution:

```python
tokens = tokenizer.encode_chat([
    {"role": "system", "content": "Answer concisely."},
    {"role": "user", "content": "What is tinygrad?"},
])
```
