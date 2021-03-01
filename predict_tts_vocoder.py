from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from utils.config_manager import Config
from data.audio import Audio
from vocoding.predictors import HiFiGANPredictor, MelGANPredictor

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config')
    parser.add_argument('--text', '-t', dest='text', default=None, type=str)
    parser.add_argument('--file', '-f', dest='file', default=None, type=str)
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--outdir', '-o', dest='outdir', default=None, type=str)
    parser.add_argument('--store_mel', '-m', dest='store_mel', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    parser.add_argument('--all_weights', '-ww', dest='all_weights', action='store_true')
    parser.add_argument('--voc', dest='vocoder', default='glim')
    parser.add_argument('--split', '-s', dest='split', action='store_true')


    args = parser.parse_args()
    
    if args.file is not None:
        with open(args.file, 'r') as file:
            text = file.readlines()
        fname = Path(args.file).stem
    elif args.text is not None:
        text = [args.text]
        fname = 'custom_text'
    else:
        fname = None
        text = None
        print(f'Specify either an input text (-t "some text") or a text input file (-f /path/to/file.txt)')
        exit()
    config_loader = Config(config_path=args.config)
    if args.outdir is None:
        outdir = config_loader.log_dir
    else:
        outdir = Path(args.outdir)
    outdir = outdir / 'outputs' / fname
    if args.vocoder == 'melgan':
        vocoder = MelGANPredictor.from_folder('vocoding/melgan/en')
    elif args.vocoder == 'hifigan':
        vocoder = HiFiGANPredictor.from_folder('vocoding/hifigan/en')
        
    outdir.mkdir(exist_ok=True, parents=True)
    audio = Audio(config_loader.config)
    if args.checkpoint is not None:
        all_weights = [args.checkpoint]
        
    elif args.all_weights:
        all_weights = [(config_loader.weights_dir / x.stem).as_posix() for x in config_loader.weights_dir.iterdir() if
                       x.suffix == '.index']
    else:
        all_weights = [None] # default
        
    print(f'\nWeights list: \n{all_weights}\n')
    for weights in all_weights:
        model = config_loader.load_model(weights)
        file_name = f'{fname}_transformer_step{model.step}'
        wavs = []
        for i, text_line in enumerate(text):
            phons = model.text_pipeline.phonemizer(text_line)
            tokens = model.text_pipeline.tokenizer(phons)
            if args.verbose:
                print(f'Predicting {text_line}')
                print(f'Phonemes: "{phons}"')
                print(f'Tokens: "{tokens}"')
            out = model.predict(tokens, encode=False)
            mel = out['mel'].numpy().T
            if args.vocoder == 'glim':
                wav = audio.reconstruct_waveform(mel)
            else:
                wav = vocoder([mel])[0]
            wavs.append(wav)
            if args.store_mel:
                np.save((outdir / (file_name + f'_{i}')).with_suffix('.mel'), out['mel'].numpy())
            if args.split:
                audio.save_wav(wav, (outdir / (file_name + f'_{i}_{args.vocoder}')).with_suffix('.wav'))
        if not args.split:
            audio.save_wav(np.concatenate(wavs), (outdir /  (file_name + f'_{args.vocoder}')).with_suffix('.wav'))
    print(f'Output wav under {outdir}')
