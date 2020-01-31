import unittest
import numpy as np
from model.transformer_utils import CharTokenizer
from preprocessing.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):

    def test_encode(self):

        tokenizer = CharTokenizer(alphabet=list('a b'))
        preprocessor = Preprocessor(mel_channels=2,
                                    start_vec_val=0.1,
                                    end_vec_val=0.2,
                                    tokenizer=tokenizer,
                                    lowercase=True,
                                    clip_val=0.1)

        mel = np.full(shape=(1, 2), fill_value=10)
        val = float(np.log(mel.clip(1e-5))[0, 0])
        mel_prep, seq, stop = preprocessor.encode('A B', mel)

        expected_mel = np.array([[0.1, 0.1], [val, val], [0.2, 0.2]])
        np.testing.assert_almost_equal(expected_mel, mel_prep)

        expected_seq = [tokenizer.start_token_index] + tokenizer.encode('a b') + [tokenizer.end_token_index]
        self.assertEqual(expected_seq, seq)

        stop_expected = np.array([1, 1, 2])
        np.testing.assert_almost_equal(stop_expected, stop)