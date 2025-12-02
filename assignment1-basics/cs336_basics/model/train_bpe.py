import os
import regex
import multiprocessing
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
from .pretokenization_example import find_chunk_boundaries

GPT2_PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def _pre_tokenize_chunk(args) -> Counter:
    """Worker function:
    read the chunk of text,calculate the frequency of bytes return the Counter """
    filepath, start, end, special_tokens = args
    counts = Counter()

    with open(filepath, "rb") as f:
        f.seek(start)
        byte_content = f.read(end - start)
    text = byte_content.decode("utf-8", errors="ignore")

    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        pattern_string = "|".join(regex.escape(s) for s in sorted_specials)
        special_pat = regex.compile(f"({pattern_string})")
        parts = special_pat.split(text)
    else:
        parts = [text]

    for part in parts:
        if not part: continue
        if special_tokens and part in special_tokens: continue

        pre_tokens = GPT2_PAT.findall(part)

        for pt in pre_tokens:
            pt_bytes = pt.encode("utf-8")
            counts[tuple(pt_bytes)] += 1

    return counts


def get_stats(vocab_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], int]:
    """
        calculate the (pair,frequency)
        e.g:
            vocab_counts:{new:5, newest:7}
            ->{(n,e):5,(e,w):5,
            (n,e):7,(e,w):7, (w,e):7, (e,s):7),(s,t):7}
            ->{(n,e):12,(e,w):12,(w,e):7, (e,s):7),(s,t):7}

            pairs = {(n,e):12,(e,w):12,(w,e):7, (e,s):7),(s,t):7}
    """
    pairs = defaultdict(int)
    for word, freq in vocab_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs


def get_pair_indices(vocab_counts: Dict[Tuple[int, ...], int]) -> Dict[Tuple[int, int], Set[Tuple[int, ...]]]:
    """
        calculate the pair and the words that contain the pair
        e.g:
            vocab_counts:{new:5, newest:7}
            ->{(n,e):5,(e,w):5,
            (n,e):7,(e,w):7, (w,e):7, (e,s):7),(s,t):7}
            ->{(n,e):12,(e,w):12,(w,e):7, (e,s):7),(s,t):7}

            pairs = {(n,e):12,(e,w):12,(w,e):7, (e,s):7),(s,t):7}
            pair_to_words = {(n,e):[new,newest],(e,w):[new,newest],(w,e):[newest],(e,s):[newest],(s,t):[newest]}
    """
    pair_to_words = defaultdict(set)
    for word in vocab_counts:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)
    return pair_to_words


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocab dict
    token_id_to_bytes = {i: bytes([i]) for i in range(256)}
    merges = []

    # fill special tokens
    next_id = 256
    for st in special_tokens:
        token_id_to_bytes[next_id] = st.encode("utf-8")
        next_id += 1

    # parallelize pre-token
    num_procs = max(1, multiprocessing.cpu_count() - 1)
    split_token = b"<|endoftext|>"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_procs, split_token)

    chunk_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk_args.append((input_path, start, end, special_tokens))

    word_counts = Counter()
    with multiprocessing.Pool(len(chunk_args)) as pool:
        for res in pool.imap_unordered(_pre_tokenize_chunk, chunk_args):
            word_counts.update(res)

    word_counts = {word: freq for word, freq in word_counts.items() if len(word) > 1}

    # --- 3. 构建统计 ---
    pair_stats = get_stats(word_counts)
    pair_index = get_pair_indices(word_counts)

    # --- 4. 训练循环 ---
    current_vocab_size = len(token_id_to_bytes)

    while current_vocab_size < vocab_size:
        if not pair_stats:
            break


        best_pair = max(
            pair_stats.items(),
            # find with pair max frequency, for those of same frequency, compare lexical order
            key=lambda x: (x[1], token_id_to_bytes[x[0][0]], token_id_to_bytes[x[0][1]])
        )

        pair_val = best_pair[0]
        count_val = best_pair[1]

        if count_val < 1: break

        # 4.2 记录合并 (存 Bytes)
        p0_bytes = token_id_to_bytes[pair_val[0]]
        p1_bytes = token_id_to_bytes[pair_val[1]]
        merges.append((p0_bytes, p1_bytes))

        # 更新词表 (存 Bytes)
        token_id_to_bytes[next_id] = p0_bytes + p1_bytes

        new_token_id = next_id
        next_id += 1
        current_vocab_size += 1

        # 4.3 增量更新
        words_to_update = list(pair_index[pair_val])
        changes = []

        for word in words_to_update:
            freq = word_counts[word]

            # 执行合并: (A, B) -> NewID
            i = 0
            new_word = []
            p0, p1 = pair_val

            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            changes.append((word, new_word, freq))

        for old_word, new_word, freq in changes:
            del word_counts[old_word]
            for i in range(len(old_word) - 1):
                p = (old_word[i], old_word[i + 1])
                pair_stats[p] -= freq
                if pair_stats[p] == 0: del pair_stats[p]
                if old_word in pair_index[p]:
                    pair_index[p].remove(old_word)
                if not pair_index[p]: del pair_index[p]

            if new_word in word_counts:
                word_counts[new_word] += freq
            else:
                word_counts[new_word] = freq

            for i in range(len(new_word) - 1):
                p = (new_word[i], new_word[i + 1])
                pair_stats[p] += freq
                pair_index[p].add(new_word)

    return token_id_to_bytes, merges