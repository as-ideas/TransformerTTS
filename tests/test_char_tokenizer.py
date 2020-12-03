import unittest

import tensorflow as tf
import numpy as np

from data.text.tokenizer import Tokenizer


class TestCharTokenizer(unittest.TestCase):
    
    def test_tokenizer(self):
        tokenizer = Tokenizer(alphabet=list('ab c'))
        self.assertEqual(5, tokenizer.start_token_index)
        self.assertEqual(6, tokenizer.end_token_index)
        self.assertEqual(7, tokenizer.vocab_size)
        
        seq = tokenizer('a b d')
        self.assertEqual([5, 1, 3, 2, 3, 6], seq)
        
        seq = np.array([5, 1, 3, 2, 8, 6])
        seq = tf.convert_to_tensor(seq)
        text = tokenizer.decode(seq)
        self.assertEqual('>a b<', text)
