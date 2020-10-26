from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils.config_manager import Config
from utils.audio import Audio

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config')
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    args = parser.parse_args()
    
    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()
    config_loader = Config(config_path=args.config, model_kind=f'forward')
    if args.outdir is None:
        outdir = config_loader.log_dir
    else:
        outdir = Path(args.outdir)
    outdir = outdir / 'outputs'
    outdir.mkdir(exist_ok=True)
    audio = Audio(config_loader.config)
    model = config_loader.load_model(args.checkpoint)
    file_name = f'{fname}_transformer_step{model.step}'
    print(f'Output wav under {(outdir / file_name).with_suffix(".wav")}')
    wavs = []
    for i, text_line in enumerate(text):
        phons = model.text_pipeline.phonemizer(text_line)
        tokens = model.text_pipeline.tokenizer(phons)
        if args.verbose:
            print(f'Predicting {text_line}')
            print(f'Phonemes: {phons}')
        out = model.predict(tokens, encode=False)
        wav = audio.reconstruct_waveform(out['mel'].numpy().T)
        wavs.append(wav)
        if args.store_mel:
            np.save((outdir / file_name + f'_{i}').with_suffix('.mel'), out['mel'].numpy())
    audio.save_wav(np.concatenate(wavs), (outdir / file_name).with_suffix('.wav'))
