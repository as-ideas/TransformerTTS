import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def buffer_mel(ms, sr):
    plt.figure(figsize=(10, 4))
    s_db = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(s_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close('all')
    return buf
