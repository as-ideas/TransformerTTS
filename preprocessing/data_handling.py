import os


def load_files(metafile,
               meldir,
               num_samples=None):
    samples = []
    count = 0
    alphabet = set()
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split('|')
            mel_file = os.path.join(str(meldir), l_split[0] + '.npy')
            text = l_split[1].strip().lower()
            phonemes = l_split[-1].strip()
            samples.append((phonemes, text, mel_file))
            alphabet.update(list(text))
            count += 1
            if num_samples is not None and count > num_samples:
                break
        alphabet = sorted(list(alphabet))
        return samples, alphabet
