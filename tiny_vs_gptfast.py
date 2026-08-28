import numpy as np 
from tinygrad import Tensor, dtypes 
from models import Transformer, load_weights

def test1():
    model = Transformer.from_name("llama-3-8b")
    load_weights(model, "checkpoints/llama-3-8b/model.pth")
    tokens = Tensor([[1,42,314,2718]], dtype=dtypes.int)
    seqlen = tokens.shape[1]
    model.setup_caches(max_batch_size=1, max_seq_length=seqlen)
    logits = model(tokens, start_pos=0).realize()
    tinygrad_logits = (logits[:,-1,:].cast(dtypes.float32).numpy())
    reference = np.load("pytorch_logits.npy") 
    print("tinygrad shape:", tinygrad_logits.shape)
    print("PyTorch shape:", reference.shape)
    print("tinygrad argmax:", tinygrad_logits.argmax(axis=-1))
    print("PyTorch argmax:", reference.argmax(axis=-1))
    diff = np.abs(tinygrad_logits - reference)
    print("max abs diff:", diff.max())
    print("mean abs diff:",diff.mean())

test1()