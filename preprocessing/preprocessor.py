import numpy as np


class Preprocessor:

    def __init__(self,
                 mel_channels,
                 start_vec_val,
                 end_vec_val,
                 tokenizer,
                 clip_val=1e-5):
        self.start_vec = np.ones((1, mel_channels)) * start_vec_val
        self.end_vec = np.ones((1, mel_channels)) * end_vec_val
        self.tokenizer = tokenizer
        self.clip_val = clip_val

    def __call__(self, sample):
        text, mel_path = sample[0], sample[1]
        text_seq = self.preprocess_text(text)
        mel = np.load(mel_path)
        mel = self.preprocess_mel(mel)
        stop_probs = np.ones((mel.shape[0]))
        stop_probs[-1] = 2
        return mel, text_seq, stop_probs

    def preprocess_mel(self, mel):
        norm_mel = np.log(mel.clip(1e-5))
        norm_mel = np.concatenate([self.start_vec, norm_mel, self.end_vec])
        return norm_mel

    def preprocess_text(self, text):
        encoded_text = self.tokenizer.encode(text)
        return [self.tokenizer.start_token_index] + encoded_text + [self.tokenizer.end_token_index]
