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
    norm_eps: float = 1e-5 
    rope_base: float = 10000.0

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

    def forward(self, x, start_pos, freqs_cis, mask=None):
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

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.w1(x).silu() * self.w3(x))

class TransformerBlock:
    def __init__(self, config):
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = nn.RMSNorm(config.dim, config.norm_eps,)
        self.ff_norm = nn.RMSNorm(config.dim, config.norm_eps,)

    def forward(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Tensor | None=None):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ff_norm(h))
        return out

class Transformer:
    def __init__(self, config: ModelArgs):
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(config.head_dim, config.block_size, config.rope_base).contiguous()
        self.max_batch_size = 0
        self.max_seq_length = 0 

    def setup_caches(self, max_batch_size: int, max_seq_length: int):
        assert max_seq_length <= self.config.block_size
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        dtype = self.output.weight.dtype 
        for layer in self.layers:
            layer.attention.kv_cache = KVCache(max_batch_size=max_batch_size, max_seq_length=max_seq_length, n_kv_heads=self.config.n_local_heads, head_dim=self.config.head_dim, dtype=dtype)
            

    def forward(self, tokens: Tensor, start_pos: int=0) -> Tensor:
        bsz, seqlen = tokens.shape
        x = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:,start_pos:start_pos+seqlen,:,:]
        if seqlen > 1: mask = Tensor.full((1,1,seqlen,start_pos+seqlen), float("-inf"), dtype=x.dtype, device=x.device,).triu(start_pos+1)
        else: mask = None 
        for layer in self.layers: x = layer.forward(x, start_pos,freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

def prefill(model: Transformer, tokens: Tensor) -> Tensor:
    logits = model(tokens, start_pos=0)
    return logits[:,-1,:]

def generate(model: Transformer, prompt: Tensor, max_new_tokens: int) -> Tensor:
    bsz, prompt_len = prompt.shape 
    assert (prompt_len + max_new_tokens <= model.config.block_size)
    model.setup_caches(max_batch_size=bsz, max_seq_length=prompt_len+max_new_tokens)
    logits = model(prompt, start_pos=0).realize()
    next_token = (logits[:,-1,:].argmax(axis=-1).reshape(bsz,1))
    generated = [next_token]
    start_pos = prompt_len
    for _ in range(max_new_tokens-1):
        logits = model(next_token, start_pos=start_pos).realize()
        next_token = (logits[:,-1,:].argmax(axis=-1).reshape(bsz,1))
        generated.append(next_token)
        start_pos += 1
    return prompt.cat(*generated, dim=1)
