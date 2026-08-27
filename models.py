from tinygrad import Tensor, nn, dtypes
from dataclasses import dataclass

@dataclass
class ModelArgs:
    block_size: int = 2048 
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32 
    dim: int = 4096 
    n_local_heads: int = -1
    head_dim: int = 0

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        self.head_dim = self.dim // self.n_head

def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x 
    return (
        x.repeat((1,1,1,n_rep)).reshape(bsz, seqlen, n_kv_heads*n_rep,head_dim)
    )

def apply_rotary_emb(q: Tensor, k: Tensor, freqa_cis: Tensor):
    q = q.reshape(*q.shape[:-1], q.shape[-1] // 2, 2)
    k = k.reshape(*k.shape[:-1], k.shape[-1] //2, 2)
    q0 = q[..., 0]
    q1 = q[..., 1]
    k0 = k[..., 0]
    k1 = k[..., 1]
    cos = freqa_cis[..., 0]
    sin = freqa_cis[..., 1]
    q_rotated = Tensor.stack(q0*cos-q1*sin, q1*cos+q0*sin,dim=-1)
    k_rotated = Tensor.stack(k0*cos-k1*sin, k1*cos+k0*sin,dim=-1)
    return (q_rotated.flatten(3), k_rotated.flatten(3),)

def precompute_freqs_cis(head_dim: int, end: int, theta:float = 10000.0) -> Tensor:
    freqs = 1.0 / (theta**(Tensor.arange(0,head_dim,2,dtype=dtypes.float32) / head_dim))
    positions = Tensor.arange(end, dtype=dtypes.float32).reshape(end,1)
    angles = positions * freqs.reshape(1,-1)
    return Tensor.stack(angles.cos(), angles.sin(), dim=-1).reshape(1,end,1,head_dim//2,2)

class Attention:
    def __init__(self, config):
        assert config.dim % config.n_head == 0
        self.dim = config.dim
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
        q, k, v = xqkv.split([self.dim, kv_size, kv_size], dim=-1) 
        q = q.reshape(bsz, seqlen, self.n_head, self.head_dim) 
        k = k.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.reshape(bsz, seqlen, self.n_local_heads, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        if self.kv_cache is not None: k, v = self.kv_cache.update(start_pos, k, v)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = q.scaled_dot_product_attention(k,v,mask)
        y = y.transpose(1,2)
        y = y.reshape(bsz, seqlen, self.dim)
        return self.wo(y)

        

def test_attention_shape():
    config = ModelArgs(dim=64,n_head=4,n_local_heads=2)
    attention = Attention(config)
    batch = 2
    seqlen = 5
    x = Tensor.randn(batch, seqlen, config.dim)
    freqs_cis = precompute_freqs_cis(config.head_dim, seqlen)
    out = attention(x,start_pos=0,freqs_cis=freqs_cis)
    print("input: ", x.shape)
    print("output: ", out.shape)
    assert out.shape == (batch, seqlen, config.dim)



if __name__ == '__main__':
    test_attention_shape()

