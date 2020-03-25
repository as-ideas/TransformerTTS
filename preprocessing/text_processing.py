from phonemizer.phonemize import phonemize

_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = sorted(list(
    _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics))
# _punctuations = '!\'(),-.:;? '
_punctuations = '!,-.:;? '
_alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäüöß'


# _alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? äüöß'


class TextCleaner:
    def __init__(self, alphabet=None):
        self.garbage = []
        if not alphabet:
            self.accepted_chars = list(_alphabet) + list(_punctuations)
    
    def clean(self, text):
        if type(text) is list:
            return [''.join([c for c in t if c in self.accepted_chars]) for t in text]
        elif type(text) is str:
            return ''.join([c for c in text if c in self.accepted_chars])
        else:
            print('Datatype not understood')
    
    def collect_garbage(self, text):
        if type(text) is list:
            out = []
            for t in text:
                clean = ''.join([c for c in t if c in self.accepted_chars])
                self.garbage.append([x for x in text if x not in clean])
                out.append(clean)
        elif type(text) is str:
            out = ''.join([c for c in t if c in self.accepted_chars])
            self.garbage.append([x for x in text if x not in out])
        else:
            print('Datatype not understood')
            out = None
        return out


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
