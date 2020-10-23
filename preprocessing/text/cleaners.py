import re
import abc
from typing import Union
from abc import abstractmethod

from preprocessing.text.symbols import _alphabet, _punctuations, _numbers
from preprocessing.text.numbers import Numbers


class Cleaner(abc.ABC):
    
    @abstractmethod
    def __call__(self, text: Union[str, list]) -> Union[str, list]:
        """ Cleans text. """
        pass


class English(Cleaner):
    def __init__(self, alphabet=None):
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations) + list(_numbers)
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
    
    def __call__(self, text: Union[str, list]) -> Union[str, list]:
        if isinstance(text, list):
            return [self._clean_line(t) for t in text]
        elif isinstance(text, str):
            return self._clean_line(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')
    
    def _get_abbreviation_pattern(self):
        return '|'.join(sorted(re.escape(k) for k in self.abbreviations))
    
    def _expand_abbreviations(self, text):
        return re.sub(self.abbreviations_pattern, lambda m: self.abbreviations.get(m.group(0)), text)
    
    def _filter_chars(self, text):
        return ''.join([c for c in text if c in self.accepted_chars])
    
    def _clean_line(self, text):
        # text = self._filter_chars(text)
        text = self._expand_numbers(text)
        text = re.sub(self.abbreviations_pattern, lambda m: self.abbreviations.get(m.group(0)), text)
        return text
    
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


class German(Cleaner):
    def __init__(self, alphabet=None):
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations) + list(_numbers)
        self.numbers = Numbers(lang_ID='de',
                               comma='Komma',
                               thousand='tausend')
        self._date_re = re.compile(r'([0-9]{1,2}\.+)')
        self._time_re = re.compile(r'([0-9]{1,2}).([0-9]{1,2})(\s*Uhr)')
    
    def __call__(self, text: Union[str, list]) -> str:
        if isinstance(text, list):
            return [self._clean_line(t) for t in text]
        elif isinstance(text, str):
            return self._clean_line(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')
    
    def _filter_chars(self, text):
        return ''.join([c for c in text if c in self.accepted_chars])
    
    def _clean_line(self, text):
        # text = self._filter_chars(text)
        text = self._expand_numbers(text)
        return text
    
    def _fix_time(self, m):
        if int(m.group(2)):
            return m.group(1) + m.group(3) + ' ' + m.group(2)  # 9 Uhr 30
        else:
            return m.group(1) + m.group(3)
    
    def _expand_date(self, m):
        num = int(m.group(0).replace('.', ''))
        if num < 20:
            return m.group(1).replace('.', 'ten')
        else:
            return m.group(1).replace('.', 'sten')
    
    def _expand_numbers(self, text):
        ends_with_dot = text.endswith('.')
        if ends_with_dot:
            text = text[:-1]
        text = self.numbers.expand_comma(text)
        text = re.sub(self._time_re, self._fix_time, text)
        text = self.numbers.expand_decimal_thousands(text)
        text = self.numbers.expand_decimal_hundreds(text)
        text = self.numbers.expand_decimal_point(text)
        text = re.sub(self._date_re, self._expand_date, text)
        text = self.numbers.expand_number(text)
        if ends_with_dot:
            text += '.'
        return text
