import regex as re

def get_stats(token_ids, stats=None):
    stats = {} if stats is None else stats
    for pair in zip(token_ids, token_ids[1:]):
        pair_count = stats.get(pair, 0) + 1
        stats[pair] = pair_count
    return stats

def merge_pairs(token_ids, given_pair, given_idx):
    new_ids = []
    i = 0
    while i<len(token_ids):
        is_not_last = i < len(token_ids) - 1
        if is_not_last and (token_ids[i], token_ids[i+1]) == given_pair:
            new_ids.append(given_idx)
            i += 2 # since the next element is merged, we skip one iteration
        else:
            new_ids.append(token_ids[i]) # we want to push same id if not paired
            i += 1 # as usual, going for next iteration
    return new_ids

#------------------- CONSTANTS 
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

#------------------- 
class BaseTokenizer:
    def __init__(self):
        self.merges = {}
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self.build_vocab() # our tokens in (int -> byte)

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError

    def decode(self, ids):
        raise NotImplementedError
    
    def build_vocab(self):
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

#-------------------

class MyTokenizer(BaseTokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.special_tokens = {} # initializing
        self.inverse_special_tokens = {} # initializing # for decoding
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text) # returns list of text chunks

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
                # find the pair with the highest count
            pair = max(stats, key=stats.get)
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge_pairs(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges # used for encode
        self.vocab = vocab # used for decode # int -> byte, build_vocab can also be used.

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text
    
    # Same as above, which we written for practice
    def _encode_chunk(self, text_bytes): # These type of functions are used only inside class by the methods (like private functions)
        # let's begin. first, convert all bytes to integers in range 0..255
        token_ids = list(text_bytes)
        for current_pair, current_idx in self.merges.items(): # merges dict must be in ascending order with respect to idx.
            token_ids = merge_pairs(token_ids, current_pair, current_idx)
        return token_ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
