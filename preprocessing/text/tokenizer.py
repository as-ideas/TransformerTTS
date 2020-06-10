from phonemizer.phonemize import phonemize

from preprocessing.text.symbols import _phonemes, _punctuations
from preprocessing.text.cleaners import English, German


class Pipeline:
    def __init__(self, language='en', add_start_end=True):
        if language == 'en':
            self.cleaner = English()
        elif language == 'de':
            self.cleaner = German()
        else:
            raise ValueError(f'language must be either "en" or "de", not {language}.')
        self.phonemizer = Phonemizer(language=language)
        self.tokenizer = Tokenizer(sorted(list(_phonemes) + list(_punctuations)), add_start_end=add_start_end)
    
    def __call__(self, input_text):
        text = self.cleaner(input_text)
        phons = self.phonemizer.encode(text)
        tokens = self.tokenizer.encode(phons)
        return tokens


class Tokenizer:
    
    def __init__(self, alphabet, start_token='>', end_token='<', pad_token='/', add_start_end=True):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.vocab_size = len(self.alphabet) + 1
        self.add_start_end = add_start_end
        if add_start_end:
            self.start_token_index = len(self.alphabet) + 1
            self.end_token_index = len(self.alphabet) + 2
            self.vocab_size += 2
            self.idx_to_token[self.start_token_index] = start_token
            self.idx_to_token[self.end_token_index] = end_token
    
    def encode(self, sentence):
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if self.add_start_end:
            sequence = [self.start_token_index] + sequence + [self.end_token_index]
        return sequence
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])


class Phonemizer:
    def __init__(self, language):
        self.language = language
    
    def encode(self, text, strip=True, preserve_punctuation=True, with_stress=False, njobs=4):
        phonemes = phonemize(text,
                             language=self.language,
                             backend='espeak',
                             strip=strip,
                             preserve_punctuation=preserve_punctuation,
                             with_stress=with_stress,
                             njobs=njobs,
                             language_switch='remove-flags')
        return phonemes
