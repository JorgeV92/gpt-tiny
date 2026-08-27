from tinygrad import Tensor, nn
from dataclasses import dataclass

@dataclass
class ModelArgs:
    block_size: int = 2048 
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32 
    dim: int = 4096 


class Attention:
    def __inti__(self, config):
        assert config.dim % config.n_head == 0
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads 
        self.head_dim = config.head_dim 
        self.n_rep = (self.n_head // self.n_local_heads)
        total_head_dim = (self.n_head + 2 * self.n_local_heads) * self.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None 

    def __call__(self, x, start_pos, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape 
        xqkv = self.wqkv(x)
        kv_size = (self.n_local_heads * self.head_dim)
        q, k, v = ... 
        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim) 
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)



def test_attention():
    pass 

