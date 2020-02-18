import numpy as np

from preprocessing.tokenizer import CharTokenizer


class Preprocessor:
    
    def __init__(self,
                 mel_channels: int,
                 start_vec_val: float,
                 end_vec_val: float,
                 tokenizer: CharTokenizer,
                 lowercase=True,
                 clip_val=1e-5):
        self.start_vec = np.ones((1, mel_channels)) * start_vec_val
        self.end_vec = np.ones((1, mel_channels)) * end_vec_val
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.mel_channels = mel_channels
        self.clip_val = clip_val
    
    def __call__(self, sample):
        text, mel_path = sample[0], sample[1]
        mel = np.load(mel_path)
        return self.encode(text, mel)
    
    def encode(self, text, mel):
        if self.lowercase:
            text = text.lower()
        encoded_text = self.tokenizer.encode(text)
        encoded_text = [self.tokenizer.start_token_index] + encoded_text + [self.tokenizer.end_token_index]
        norm_mel = np.log(mel.clip(1e-5))
        norm_mel = np.concatenate([self.start_vec, norm_mel, self.end_vec], axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[-1] = 2
        return norm_mel, encoded_text, stop_probs
