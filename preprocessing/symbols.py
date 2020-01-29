""" from https://github.com/keithito/tacotron """

_pad = '_'
_punctuation = '!\'(),.:;?" '
_special = '-'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÜÖabcdefghijklmnopqrstuvwxyzäüöß'

# Export all symbols:
alphabet = [_pad] + list(_special) + list(_punctuation) + list(_letters)
