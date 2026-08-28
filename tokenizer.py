from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

from tinygrad import Tensor, dtypes


@runtime_checkable
class TextTokenizer(Protocol):
    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def stop_token_ids(self) -> frozenset[int]: ...

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str,*,add_bos: bool = False,add_eos: bool = False, allow_special: bool = False,) -> list[int]: ...
    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str: ...


class ChatMessage(TypedDict):
    role: str
    content: str


def _load_tiktoken():
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Llama3Tokenizer requires tiktoken. Install project dependencies "
            "with `python -m pip install -r requirements.txt`."
        ) from exc
    return tiktoken


def _llama3_special_token_names() -> tuple[str, ...]:
    leading = (
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",
    )
    remaining = tuple(f"<|reserved_special_token_{index}|>" for index in range(5, 251))
    names = leading + remaining
    assert len(names) == 256 and len(set(names)) == 256
    return names


class Llama3Tokenizer:
    _PATTERN = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
        r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )

    def __init__(self, encoding, special_tokens: Mapping[str, int], base_vocab_size: int):
        self._encoding = encoding
        self._special_tokens = dict(special_tokens)
        self._special_ids = frozenset(self._special_tokens.values())
        self._base_vocab_size = base_vocab_size

    @classmethod
    def from_file(cls, model_path: str | Path) -> Llama3Tokenizer:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"tokenizer rank file not found: {path}")
        _load_tiktoken()
        from tiktoken.load import load_tiktoken_bpe
        return cls.from_mergeable_ranks(load_tiktoken_bpe(str(path)), name=path.name)

    @classmethod
    def from_mergeable_ranks(cls,mergeable_ranks: Mapping[bytes, int],*,name: str = "gpt-tiny-llama3",) -> Llama3Tokenizer:
        tiktoken = _load_tiktoken()
        ranks = dict(mergeable_ranks)
        if not ranks:
            raise ValueError("mergeable_ranks cannot be empty")

        base_vocab_size = len(ranks)
        special_tokens = { token: base_vocab_size + offset for offset, token in enumerate(_llama3_special_token_names())}
        encoding = tiktoken.Encoding(name=name,pat_str=cls._PATTERN,mergeable_ranks=ranks,special_tokens=special_tokens,)
        return cls(encoding, special_tokens, base_vocab_size)

    @property
    def bos_id(self) -> int:
        return self._special_tokens["<|begin_of_text|>"]

    @property
    def eos_id(self) -> int:
        return self._special_tokens["<|end_of_text|>"]

    @property
    def eot_id(self) -> int:
        return self._special_tokens["<|eot_id|>"]

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return frozenset((self.eos_id, self.eot_id))

    @property
    def vocab_size(self) -> int:
        return self._base_vocab_size + len(self._special_tokens)

    def special_token_id(self, token: str) -> int:
        try:
            return self._special_tokens[token]
        except KeyError as exc:
            raise ValueError(f"unknown Llama 3 special token: {token}") from exc

    def encode(self,text: str,*,add_bos: bool = False,add_eos: bool = False,allow_special: bool = False) -> list[int]:
        if not isinstance(text, str): raise TypeError("text must be a string")
        allowed_special = "all" if allow_special else set()
        encoded = self._encoding.encode(text,allowed_special=allowed_special,disallowed_special=())
        if add_bos: encoded.insert(0, self.bos_id)
        if add_eos: encoded.append(self.eos_id)
        return encoded

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if skip_special: ids = [token_id for token_id in ids if token_id not in self._special_ids]
        return self._encoding.decode(ids)

    def encode_chat(self,messages: Sequence[ChatMessage],*,add_generation_prompt: bool = True,) -> list[int]:
        tokens = [self.bos_id]
        start_header = self.special_token_id("<|start_header_id|>")
        end_header = self.special_token_id("<|end_header_id|>")
        for message in messages:
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or not role:
                raise ValueError("each chat message needs a non-empty string role")
            if not isinstance(content, str):
                raise ValueError("each chat message needs string content")

            tokens.append(start_header)
            tokens.extend(self.encode(role))
            tokens.append(end_header)
            tokens.extend(self.encode("\n\n" + content.strip()))
            tokens.append(self.eot_id)

        if add_generation_prompt:
            tokens.append(start_header)
            tokens.extend(self.encode("assistant"))
            tokens.append(end_header)
            tokens.extend(self.encode("\n\n"))
        return tokens


class ByteTokenizer:
    _BOS = 256
    _EOS = 257
    _EOT = 258

    @property
    def bos_id(self) -> int:
        return self._BOS

    @property
    def eos_id(self) -> int:
        return self._EOS

    @property
    def stop_token_ids(self) -> frozenset[int]:
        return frozenset((self._EOS, self._EOT))

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self,text: str,*,add_bos: bool = False,add_eos: bool = False,allow_special: bool = False,) -> list[int]:
        del allow_special
        if not isinstance(text, str): raise TypeError("text must be a string")
        encoded = list(text.encode("utf-8"))
        if add_bos: encoded.insert(0, self.bos_id)
        if add_eos: encoded.append(self.eos_id)
        return encoded

    def decode(self, token_ids: Iterable[int], *, skip_special: bool = True) -> str:
        values: list[int] = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id >= 256:
                if skip_special and token_id < self.vocab_size:
                    continue
                raise ValueError(f"byte tokenizer cannot decode token {token_id}")
            if token_id < 0:
                raise ValueError(f"byte tokenizer cannot decode token {token_id}")
            values.append(token_id)
        return bytes(values).decode("utf-8", errors="replace")


def encode_tensor(tokenizer: TextTokenizer,text: str,*,add_bos: bool = True,add_eos: bool = False,device: str | None = None) -> Tensor:
    """Encode one prompt as a batched tinygrad integer tensor."""
    token_ids = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    if not token_ids: raise ValueError("the encoded prompt is empty")
    tensor = Tensor([token_ids], dtype=dtypes.int)
    return tensor if device is None else tensor.to(device)

