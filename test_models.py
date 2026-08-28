from tinygrad import Tensor, dtypes

from models import (
    Attention,
    FeedForward,
    KVCache,
    ModelArgs,
    Transformer,
    TransformerBlock,
    apply_rotary_emb,
    precompute_freqs_cis,
    repeat_kv,
    generate,
)


def test_attention_shape():
    config = ModelArgs(dim=64, n_head=4, n_local_heads=2)
    attention = Attention(config)
    batch = 2
    seqlen = 5
    x = Tensor.randn(batch, seqlen, config.dim)
    freqs_cis = precompute_freqs_cis(config.head_dim, seqlen)
    out = attention.forward(x, start_pos=0, freqs_cis=freqs_cis)
    print("input: ", x.shape)
    print("output: ", out.shape)
    assert out.shape == (batch, seqlen, config.dim)


def test_repeat_kv():
    x = Tensor.arange(1 * 2 * 2 * 4).reshape(1, 2, 2, 4)
    res = repeat_kv(x, 2)
    print("before :", x.shape)
    print("after :", res.shape)
    assert res.shape == (1, 2, 4, 4)


def test_rope_preserves_norm():
    q = Tensor.randn(1, 4, 4, 16)
    k = Tensor.randn(1, 4, 2, 16)
    freqs = precompute_freqs_cis(head_dim=16, end=4)
    q_rot, k_rot = apply_rotary_emb(q, k, freqs)
    before = (q * q).sum(axis=-1)
    after = (q_rot * q_rot).sum(axis=-1)
    assert before.allclose(after, rtol=1e-4, atol=1e-4).item()


def test_kv_cache():
    batch = 1
    max_seq_len = 8
    n_kv_heads = 2
    head_dim = 4
    cache = KVCache(
        max_batch_size=batch,
        max_seq_length=max_seq_len,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dtype=dtypes.float32,
    )
    k1 = Tensor.arange(
        batch * 3 * n_kv_heads * head_dim, dtype=dtypes.float32
    ).reshape(batch, 3, n_kv_heads, head_dim)
    v1 = k1 + 100
    keys, values = cache.update(start_pos=0, k=k1, v=v1)
    print("after prefill:")
    print("keys shape: ", keys.shape)
    print("values shape: ", values.shape)
    assert keys.shape == (batch, 3, n_kv_heads, head_dim)
    assert values.shape == (batch, 3, n_kv_heads, head_dim)
    assert keys.allclose(k1).item()
    assert values.allclose(v1).item()
    k2 = Tensor.full((batch, 1, n_kv_heads, head_dim), 999.0)
    v2 = Tensor.full((batch, 1, n_kv_heads, head_dim), 1999.0)
    keys, values = cache.update(start_pos=3, k=k2, v=v2)
    print("after decode:")
    print("keys shape   :", keys.shape)
    print("values shape :", values.shape)
    assert keys.shape == (batch, 4, n_kv_heads, head_dim)
    assert values.shape == (batch, 4, n_kv_heads, head_dim)
    assert keys[:, :3].allclose(k1).item()
    assert values[:, :3].allclose(v1).item()
    assert keys[:, 3:4].allclose(k2).item()
    assert values[:, 3:4].allclose(v2).item()
    print("KVCache test passed")


def test_feed_forward_shape():
    config = ModelArgs(dim=64, n_head=4, n_local_heads=2)
    ff = FeedForward(config)
    x = Tensor.randn(2, 5, 64)
    out = ff.forward(x)
    print("input: ", x.shape)
    print("output: ", out.shape)
    assert out.shape == x.shape
    print("FeedForward test passed")


def test_transformer_block_shape():
    config = ModelArgs(dim=64, n_head=4, n_local_heads=2)
    block = TransformerBlock(config)
    batch = 2
    seqlen = 5
    x = Tensor.randn(batch, seqlen, config.dim)
    freqs_cis = precompute_freqs_cis(config.head_dim, seqlen)
    out = block.forward(x, start_pos=0, freqs_cis=freqs_cis)
    print("input: ", x.shape)
    print("output: ", out.shape)
    assert out.shape == x.shape
    print("TransformerBlock test passed")


def test_transformer_shape():
    config = ModelArgs(
        block_size=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_local_heads=2,
        dim=64,
    )
    model = Transformer(config)
    tokens = Tensor([[1, 5, 20, 17, 9], [4, 2, 11, 8, 6]])
    logits = model.forward(tokens, start_pos=0)
    print("tokens shape: ", tokens.shape)
    print("logits shape: ", logits.shape)
    assert tokens.shape == (2, 5)
    assert logits.shape == (2, 5, config.vocab_size)
    print("Transformer test passed")


def test_generation():
    config = ModelArgs(block_size=32, vocab_size=128,n_layer=2,n_head=4,n_local_heads=2,dim=64)
    model = Transformer(config)
    prompt = Tensor([[1,5,20,17,9]])
    result = generate(model,prompt,max_new_tokens=5)
    print("prompt shape: ", prompt.shape)
    print("result shape: ", result.shape)
    print(result.numpy())
    assert result.shape == (1,10)
    print("Generation test passed")


def run_tests():
    test_attention_shape()
    test_repeat_kv()
    test_rope_preserves_norm()
    test_kv_cache()
    test_feed_forward_shape()
    test_transformer_block_shape()
    test_transformer_shape()


if __name__ == "__main__":
    run_tests()
