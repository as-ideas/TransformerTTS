import re

from num2words import num2words

# TODO: add first second third and cardinals
class Numbers:
    def __init__(self, lang_ID, comma, thousand):
        self.lang_ID = lang_ID
        self.comma = comma
        self.thousand = thousand
        self._comma_number_re = re.compile(r'([0-9]+,[0-9]+)')
        self._decimal_number_re = re.compile(
            r'(\d+\.\d{1,2}[^.\d])')  # excludes those digits that have a double dot eg dates 24.03.
        self._number_re = re.compile(r'[0-9]+')
        self._decimal_thousands_re = re.compile(r'(\.000)')
        self._decimal_hundreds_re = re.compile(r'(\.\d\d\d)')
    
    def _expand_comma(self, m):
        return m.group(1).replace(',', f' {self.comma} ')
    
    def _expand_decimal_point(self, m):
        return m.group(1).replace('.', f' {self.comma} ')
    
    def _expand_decimal_thousands(self, m):
        return m.group(1).replace('.000', f'{self.thousand}')
    
    def _expand_decimal_hundreds(self, m):
        return m.group(1).replace('.', f'{self.thousand}')
    
    def _expand_number(self, m):
        num = int(m.group(0))
        return num2words(num, lang=self.lang_ID)
    
    def expand_comma(self, text):
        return re.sub(self._comma_number_re, self._expand_comma, text)
    
    def expand_decimal_thousands(self, text):
        return re.sub(self._decimal_thousands_re, self._expand_decimal_thousands, text)
    
    def expand_decimal_hundreds(self, text):
        return re.sub(self._decimal_hundreds_re, self._expand_decimal_hundreds, text)
    
    def expand_decimal_point(self, text):
        return re.sub(self._decimal_number_re, self._expand_decimal_point, text)
    
    def expand_number(self, text):
        return re.sub(self._number_re, self._expand_number, text)
