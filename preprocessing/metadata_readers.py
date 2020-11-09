"""
    methods for reading a dataset and return a dictionary of the form:
    {
      filename: text_line,
      ...
    }
"""

import sys
from typing import Dict, List, Tuple

def get_preprocessor_by_name(name: str):
    """
    Returns the respective preprocessing function.
    Taken from https://github.com/mozilla/TTS/blob/master/TTS/tts/datasets/preprocess.py
    """
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def ljspeech(metadata_path: str, column_sep='|') -> dict:
    text_dict = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split(column_sep)
            filename, text = l_split[0], l_split[-1]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            text = text.replace('\n', '')
            text_dict.update({filename: text})
    return text_dict


def asvoice2(metadata_path: str, column_sep='|') -> dict:
    text_dict = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split(column_sep)
            filename, text = l_split[0], l_split[1]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            text = text.replace('\n', '')
            text_dict.update({filename: text})
    return text_dict


def post_processed_reader(metadata_path: str, column_sep='|', upsample_indicators='?!', upsample_factor=10) -> Tuple[Dict, List]:
    """
    Used to read metadata files created within the repo.
    """
    text_dict = {}
    upsample = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split(column_sep)
            filename, text = l_split[0], l_split[1]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            text = text.replace('\n', '')
            if any(el in text for el in list(upsample_indicators)):
                upsample.extend([filename]*upsample_factor)
            text_dict.update({filename: text})
    return text_dict, upsample


if __name__ == '__main__':
    metadata_path = '/Volumes/data/datasets/LJSpeech-1.1/metadata.csv'
    d = get_preprocessor_by_name('ljspeech')(metadata_path)
    key_list = list(d.keys())
    print('metadata head')
    for key in key_list[:5]:
        print(f'{key}: {d[key]}')
    print('metadata tail')
    for key in key_list[-5:]:
        print(f'{key}: {d[key]}')
