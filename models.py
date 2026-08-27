from tinygrad import Tensor, nn, dtypes
from dataclasses import dataclass

def find_mult(n: int, k: int) -> int:
    if n % k == 0: 
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048 
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32 
    dim: int = 4096 
    n_local_heads: int = -1
    head_dim: int = 0
    intermediate_size: int | None = None 

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        self.head_dim = self.dim // self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim 
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_mult(n_hidden, 256)

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

class KVCache:
    def __init__(self, max_batch_size: int, max_seq_length: int, n_kv_heads: int, head_dim: int, dtype=dtypes.float32):
        self.k_cache = Tensor.zeros(max_batch_size, max_seq_length, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()
        self.v_cache = Tensor.zeros(max_batch_size, max_seq_length, n_kv_heads, head_dim, dtype=dtype).contiguous().realize()

    def update(self, start_pos: int, k: Tensor, v: Tensor): 
        bsz, seqlen, _, _ = k.shape 
        end_pos = start_pos + seqlen 
        self.k_cache[:bsz, start_pos:end_pos,:,:].assign(k).realize()
        self.v_cache[:bsz, start_pos:end_pos, :, :].assign(v).realize()
        keys = self.k_cache[:bsz, :end_pos, :,:]
        values = self.v_cache[:bsz,:end_pos,:,:]
        return keys, values 
        
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

class FeedForward: 
    def __init__(self, config):
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))

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

def test_repeat_kv():
    x = Tensor.arange(1*2*2*4).reshape(1,2,2,4)
    res = repeat_kv(x, 2)
    print("before :", x.shape)
    print("after :", res.shape)
    assert res.shape == (1,2,4,4)

def test_rope_preserves_norm():
    q = Tensor.randn(1,4,4,16,)
    k = Tensor.randn(1,4,2,16,)
    freqs = precompute_freqs_cis(head_dim=16,end=4)
    q_rot, k_rot = apply_rotary_emb(q, k,freqs)
    before = (q*q).sum(axis=-1)
    after = (q_rot*q_rot).sum(axis=-1)
    assert before.allclose(after, rtol=1e-4,atol=1e-4).item()

def test_kv_cache():
    batch = 1
    max_seq_len = 8 
    n_kv_heads = 2
    head_dim = 4 
    cache = KVCache(max_batch_size=batch, max_seq_length=max_seq_len, n_kv_heads=n_kv_heads, head_dim=head_dim, dtype=dtypes.float32)
    k1 = Tensor.arange(batch*3*n_kv_heads*head_dim, dtype=dtypes.float32,).reshape(batch,3,n_kv_heads,head_dim)
    v1 = k1 + 100 
    keys, values = cache.update(start_pos=0,k=k1,v=v1)
    print("after prefill:")
    print("keys shape: ", keys.shape)
    print("values shape: ", values.shape)
    assert keys.shape == (batch, 3, n_kv_heads, head_dim, )
    assert values.shape == (batch, 3, n_kv_heads, head_dim)
    assert keys.allclose(k1).item()
    assert values.allclose(v1).item()
    k2 = Tensor.full((batch, 1, n_kv_heads, head_dim), 999.0)
    v2 = Tensor.full((batch,1,n_kv_heads,head_dim), 1999.0)
    keys, values = cache.update(start_pos=3, k=k2, v=v2)
    print("after decode:")
    print("keys shape   :", keys.shape)
    print("values shape :", values.shape)
    assert keys.shape == (batch, 4, n_kv_heads, head_dim)
    assert values.shape == (batch, 4, n_kv_heads, head_dim)
    assert keys[:, :3].allclose(k1).item()
    assert values[:,:3].allclose(v1).item()
    assert keys[:, 3:4].allclose(k2).item()
    assert values[:, 3:4].allclose(v2).item()
    print("KVCache test passed")

def test_feed_forward_shape():
    config = ModelArgs(dim=64, n_head=4, n_local_heads=2)
    ff = FeedForward(config)
    x = Tensor.randn(2,5,64,)
    out = ff(x)
    print("input: ", x.shape)
    print("output: ", out.shape)
    assert out.shape == x.shape
    print("FeedForward test passed")

if __name__ == '__main__':
    test_attention_shape()
    test_repeat_kv()
    test_rope_preserves_norm()
    test_kv_cache()
    test_feed_forward_shape()
