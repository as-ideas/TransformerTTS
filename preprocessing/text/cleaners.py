import re

from preprocessing.text.symbols import _alphabet, _punctuations, _not_end_punctuation, _numbers
from preprocessing.text.numbers import Numbers


class Cleaner:
    def __init__(self, alphabet=None):
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations) + list(_numbers)
        self.abbreviations = {}
        self.abbreviations_pattern = None
    
    def __call__(self, text):
        if type(text) is list:
            return [self._clean_line(t) for t in text]
        elif type(text) is str:
            return self._clean_line(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')
    
    def _get_abbreviation_pattern(self):
        return '|'.join(sorted(re.escape(k) for k in self.abbreviations))
    
    def _expand_numbers(self, text):
        raise NotImplementedError
    
    def _filter_chars(self, text):
        return ''.join([c for c in text if c in self.accepted_chars])
    
    def _clean_line(self, text):
        text = self._filter_chars(text)
        text = self._expand_numbers(text)
        if self.abbreviations:
            text = re.sub(self.abbreviations_pattern, lambda m: self.abbreviations.get(m.group(0)), text)
        if text.endswith(tuple(_not_end_punctuation)):
            text = text[:-1]
        return text + ' '


class English(Cleaner):
    def __init__(self):
        super().__init__()
        self.numbers = Numbers(lang_ID='en',
                               comma='comma',
                               thousand='thousands')
        self.abbreviations = {
            'Mrs.': 'Mrs',
            'Mr.': 'Mr',
            'Dr.': 'Dr',
            'St.': 'St',
            'Co.': 'Co',
            'Jr.': 'Jr',
            'Maj.': 'Maj',
            'Gen.': 'Gen',
            'Drs.': 'Drs',
            'Rev.': 'Rev',
            'Lt.': 'Lt',
            'Hon.': 'Hon',
            'Sgt.': 'Sgt',
            'Capt.': 'Capt',
            'Esq.': 'Esq',
            'Ltd.': 'Ltd',
            'Col.': 'Col',
            'Ft.': 'Ft',
            'a.m.': 'a m',
            'p.m.': 'p m',
            'e.g.': 'e g',
            'i.e.': 'i e',
            ';': ',',
            ':': ','}
        
        self.abbreviations_pattern = self._get_abbreviation_pattern()
    
    def _expand_abbreviations(self, text):
        return re.sub(self.abbreviations_pattern, lambda m: self.abbreviations.get(m.group(0)), text)
    
    def _expand_numbers(self, text):
        ends_with_dot = text.endswith('.')
        if ends_with_dot:
            text = text[:-1]
        text = self.numbers.expand_comma(text)
        text = self.numbers.expand_decimal_thousands(text)
        text = self.numbers.expand_decimal_hundreds(text)
        text = self.numbers.expand_decimal_point(text)
        text = self.numbers.expand_number(text)
        if ends_with_dot:
            text += '.'
        return text
