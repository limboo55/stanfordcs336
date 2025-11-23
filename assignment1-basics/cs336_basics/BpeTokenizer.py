from heapq import merge
from typing import Iterable

import regex


class BpeTokenizer:

    def __init__(self, vocab : dict[int, bytes],
                 merges: list[tuple[bytes,bytes]], special_tokens : list[str] | None = None):

        self.vocab_id_to_token : dict[int, bytes] = vocab
        self.token_to_vocab_id : dict[bytes, int] = {token:_id for _id, token in vocab.items()}
        #  GPT-2 regex
        self.pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.merges_ranks: dict[tuple[bytes,bytes], int] = {pair:rank for rank,pair in enumerate(merges)}


        # handle special tokens
        self.special_tokens = special_tokens if special_tokens else []
        if self.special_tokens:
            self.special_tokens = sorted(self.special_tokens,key = len, reverse= True)
            pattern_string = "|".join(regex.escape(s) for s in self.special_tokens)
            self.special_pat = regex.compile(f"({pattern_string})")
        else:
            self.special_pat = None

        next_id = max(self.vocab_id_to_token.keys()) + 1

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("UTF-8")
            if special_bytes not in self.token_to_vocab_id:
                self.token_to_vocab_id[special_bytes] = next_id
                self.vocab_id_to_token[next_id] = special_bytes
                next_id += 1

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens: list[str] | None = None):
        """
        Class method to load tokenizer from disk files.
        """
        import json

        vocab: dict[int, bytes] = {}

        try:
            with open(vocab_filepath, "r", encoding="utf-8") as f:

                raw_vocab = json.load(f)

            for str_id, token_str in raw_vocab.items():

                token_id = int(str_id)
                token_bytes = token_str.encode("latin-1")

                vocab[token_id] = token_bytes

        except FileNotFoundError:
            raise FileNotFoundError(f"Vocab file not found at: {vocab_filepath}")


        merges: list[tuple[bytes, bytes]] = []

        try:
            with open(merges_filepath, "r", encoding="utf-8") as f:
                for line in f:

                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split(" ")
                    if len(parts) == 2:
                        token1 = parts[0].encode("latin-1")
                        token2 = parts[1].encode("latin-1")
                        merges.append((token1, token2))

        except FileNotFoundError:
            raise FileNotFoundError(f"Merges file not found at: {merges_filepath}")


        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        encoded : list[int]= []
        if self.special_pat:
            parts = self.special_pat.split(text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                part_encode = part.encode("UTF-8")
                encoded.append(self.token_to_vocab_id[part_encode])
            else:
                part_split = self.pat.findall(part)
                for pre_token in part_split:
                    pre_token_encode = pre_token.encode("UTF-8")
                    byte_list : list[bytes]= [bytes([b]) for b in pre_token_encode]

                    byte_list_merged = self._apply_merges_to_tokens(byte_list)
                    encoded += [self.token_to_vocab_id[item] for item in byte_list_merged ]
        return encoded


    def encode_iterable(self, iterable : Iterable[str]) -> Iterable[int]:
        for text in iterable:
            token = self.encode(text)
            yield from token

    def decode(self, ids: list[int]) -> str:
        byte_list: list[bytes] = [self.vocab_id_to_token[_id] for _id in ids]
        byte_string = b"".join(byte_list)
        res_string = byte_string.decode(encoding="UTF-8",errors="replace")
        return res_string

    def _apply_merges_to_tokens(self,byte_list: list[bytes]) -> list[bytes]:

        while True:
            highest_pair = None
            highest_pair_idx = -1
            highest_rank = float('inf')
            for idx in range(len(byte_list) - 1):
                pair = (byte_list[idx],byte_list[idx + 1])

                if pair in self.merges_ranks:
                    if self.merges_ranks[pair] < highest_rank:
                        highest_pair = pair
                        highest_rank = self.merges_ranks[pair]
                        highest_pair_idx = idx
            if highest_pair is None:
                break
            byte_list = byte_list[:highest_pair_idx] + \
                        [byte_list[highest_pair_idx] + byte_list[highest_pair_idx + 1]] + \
                        byte_list[highest_pair_idx + 2 :]

        return byte_list


