from preprocessing.text.symbols import all_phonemes
from preprocessing.text.tokenizer import Phonemizer, Tokenizer


class TextToTokens:
    def __init__(self, phonemizer: Phonemizer, tokenizer: Tokenizer):
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer
    
    def __call__(self, input_text):
        phons = self.phonemizer(input_text)
        tokens = self.tokenizer(phons)
        return tokens
    
    @classmethod
    def default(cls, language, add_start_end, with_stress, njobs=1):
        phonemizer = Phonemizer(language=language, njobs=njobs, with_stress=with_stress)
        tokenizer = Tokenizer(add_start_end=add_start_end)
        return cls(phonemizer=phonemizer, tokenizer=tokenizer)