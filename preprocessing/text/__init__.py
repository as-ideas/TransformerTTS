from preprocessing.text.cleaners import English, German
from preprocessing.text.symbols import _phonemes, _punctuations
from preprocessing.text.tokenizer import Phonemizer, Tokenizer


class Pipeline:
    def __init__(self, cleaner, phonemizer, tokenizer):
        self.cleaner = cleaner
        self.phonemizer = phonemizer
        self.tokenizer = tokenizer
    
    def __call__(self, input_text):
        text = self.cleaner(input_text)
        phons = self.phonemizer(text)
        tokens = self.tokenizer(phons)
        return tokens
    
    @classmethod
    def default_pipeline(cls, language, add_start_end):
        if language == 'en':
            cleaner = English()
        elif language == 'de':
            cleaner = German()
        else:
            raise ValueError(f'language must be either "en" or "de", not {language}.')
        phonemizer = Phonemizer(language=language, strip=False, njobs=1)
        tokenizer = Tokenizer(sorted(list(_phonemes) + list(_punctuations)), add_start_end=add_start_end)
        return cls(cleaner=cleaner, phonemizer=phonemizer, tokenizer=tokenizer)
    
    @classmethod
    def default_training_pipeline(cls, language, add_start_end):
        if language == 'en':
            cleaner = English()
        elif language == 'de':
            cleaner = German()
        else:
            raise ValueError(f'language must be either "en" or "de", not {language}.')
        phonemizer = Phonemizer(language=language, strip=True, njobs=4)
        tokenizer = Tokenizer(sorted(list(_phonemes) + list(_punctuations)), add_start_end=add_start_end)
        return cls(cleaner=cleaner, phonemizer=phonemizer, tokenizer=tokenizer)
