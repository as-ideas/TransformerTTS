class Tokenizer:
    
    def __init__(self, alphabet, start_token='>', end_token='<', pad_token='/'):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet) + 1
        self.end_token_index = len(self.alphabet) + 2
        self.vocab_size = len(self.alphabet) + 3
        self.idx_to_token[self.start_token_index] = start_token
        self.idx_to_token[self.end_token_index] = end_token
    
    def encode(self, sentence):
        return [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])