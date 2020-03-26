import unittest

import numpy as np

from utils.losses import new_scaled_crossentropy, masked_crossentropy


class TestCharTokenizer(unittest.TestCase):
    
    def test_crossentropy(self):
        scaled_crossent = new_scaled_crossentropy(index=2, scaling=5)
        
        targets = np.array([[0, 1, 2]])
        logits = np.array([[[.3, .2, .1], [.3, .2, .1], [.3, .2, .1]]])
        
        loss = scaled_crossent(targets, logits)
        self.assertAlmostEqual(2.3705523014068604, float(loss))
        
        scaled_crossent = new_scaled_crossentropy(index=2, scaling=1)
        loss = scaled_crossent(targets, logits)
        self.assertAlmostEqual(0.7679619193077087, float(loss))
        
        loss = masked_crossentropy(targets, logits)
        self.assertAlmostEqual(0.7679619193077087, float(loss))
