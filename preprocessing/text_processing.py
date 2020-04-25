import re

from phonemizer.phonemize import phonemize

_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(
    _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))
_punctuations = '!,-.:;? '
_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäüöß'
_not_end_punctuation = ',-.:; '
_ad_hoc_replace = {
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
_ad_hoc_pattern = '|'.join(sorted(re.escape(k) for k in _ad_hoc_replace))


class TextCleaner:
    def __init__(self, alphabet=None):
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations)
    
    def clean(self, text):
        if type(text) is list:
            return [self.clean_line(t) for t in text]
        elif type(text) is str:
            return self.clean_line(text)
        else:
            raise TypeError(f'TextCleaner.clean() input must be list or str, not {type(text)}')
    
    def clean_line(self, text):
        text = ''.join([c for c in text if c in self.accepted_chars])
        text = re.sub(_ad_hoc_pattern, lambda m: _ad_hoc_replace.get(m.group(0)), text)
        if text.endswith(tuple(_not_end_punctuation)):
            text = text[:-1]
        return text + ' '


class Phonemizer:
    def __init__(self, language, alphabet=None):
        self.language = language
        self.cleaner = TextCleaner(alphabet)
    
    def encode(self, text, strip=True, preserve_punctuation=True, with_stress=False, njobs=4, clean=True):
        if clean:
            text = self.cleaner.clean(text)
        phonemes = phonemize(text,
                             language=self.language,
                             backend='espeak',
                             strip=strip,
                             preserve_punctuation=preserve_punctuation,
                             with_stress=with_stress,
                             njobs=njobs,
                             language_switch='remove-flags')
        return phonemes
