# from typing import Union
#
# import numpy as np
#
# from preprocessing.data_handling  import Tokenizer#, PhonemeTokenizer
#
#
# class DataPrepper:
#
#     def __init__(self,
#                  mel_channels: int,
#                  start_vec_val: float,
#                  end_vec_val: float,
#                  tokenizer: Union[Tokenizer],
#                  lowercase=True,
#                  mel_clip_val=1e-5):
#         self.start_vec = np.ones((1, mel_channels)) * start_vec_val
#         self.end_vec = np.ones((1, mel_channels)) * end_vec_val
#         self.tokenizer = tokenizer
#         self.lowercase = lowercase
#         self.mel_channels = mel_channels
#         self.mel_clip_val = mel_clip_val
#
#     def __call__(self, sample, include_text=True):
#         phonemes, text, mel_path = sample
#         mel = np.load(mel_path)
#         return self.__run(phonemes, text, mel, include_text=include_text)
#
#     def __run(self, phonemes, text, mel, include_text):
#         if self.lowercase:
#             phonemes = phonemes.lower()
#         encoded_phonemes = self.tokenizer.encode(phonemes)
#         encoded_phonemes = [self.tokenizer.start_token_index] + encoded_phonemes + [self.tokenizer.end_token_index]
#         norm_mel = np.log(mel.clip(self.mel_clip_val))
#         norm_mel = np.concatenate([self.start_vec, norm_mel, self.end_vec], axis=0)
#         stop_probs = np.ones((norm_mel.shape[0]))
#         stop_probs[-1] = 2
#         if include_text:
#             return norm_mel, encoded_phonemes, stop_probs, text
#         else:
#             return norm_mel, encoded_phonemes, stop_probs
#
#
