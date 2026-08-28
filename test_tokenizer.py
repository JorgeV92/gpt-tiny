import base64

import pytest

from tokenizer import ByteTokenizer, Llama3Tokenizer, encode_tensor


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "tinygrad says hello π"
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id
    assert tokenizer.decode(tokens) == text
    assert tokenizer.vocab_size == 259


def test_encode_tensor_is_batched():
    tokenizer = ByteTokenizer()
    tokens = encode_tensor(tokenizer, "hi")

    assert tokens.shape == (1, 3)
    assert tokens.tolist() == [[tokenizer.bos_id, ord("h"), ord("i")]]


def test_byte_tokenizer_rejects_unknown_token():
    with pytest.raises(ValueError, match="cannot decode"):
        ByteTokenizer().decode([999])


def test_llama3_tokenizer_round_trip_without_model_download():
    pytest.importorskip("tiktoken")
    ranks = {bytes((value,)): value for value in range(256)}
    tokenizer = Llama3Tokenizer.from_mergeable_ranks(ranks, name="gpt-tiny-test")
    text = "Tokenizer smoke test: λ"

    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)

    assert tokenizer.vocab_size == 512
    assert tokens[0] == tokenizer.bos_id
    assert tokens[-1] == tokenizer.eos_id
    assert tokenizer.decode(tokens) == text


def test_llama3_tokenizer_loads_local_rank_file(tmp_path):
    pytest.importorskip("tiktoken")
    rank_file = tmp_path / "tokenizer.model"
    lines = [
        base64.b64encode(bytes((value,))).decode("ascii") + f" {value}"
        for value in range(256)
    ]
    rank_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tokenizer = Llama3Tokenizer.from_file(rank_file)

    assert tokenizer.decode(tokenizer.encode("local tokenizer")) == "local tokenizer"


def test_llama3_special_tokens_and_chat_format():
    pytest.importorskip("tiktoken")
    ranks = {bytes((value,)): value for value in range(256)}
    tokenizer = Llama3Tokenizer.from_mergeable_ranks(ranks, name="gpt-tiny-chat-test")

    encoded_eot = tokenizer.encode("<|eot_id|>", allow_special=True)
    chat = tokenizer.encode_chat([{"role": "user", "content": " Hello "}])

    assert encoded_eot == [tokenizer.eot_id]
    assert chat[0] == tokenizer.bos_id
    assert tokenizer.eot_id in chat
    assert chat[-1] != tokenizer.eot_id


def test_llama3_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer rank file not found"):
        Llama3Tokenizer.from_file(tmp_path / "tokenizer.model")
