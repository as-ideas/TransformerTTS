
# for display mel
import librosa
import librosa.display
import matplotlib.pyplot as plt


def display_mel(ms, sr, file=None):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    if not file:
        plt.show()
    else:
        plt.savefig(file)
    plt.close('all')

