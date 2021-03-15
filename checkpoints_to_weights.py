from argparse import ArgumentParser

from utils.config_manager import Config

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', dest='config')
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', default=None, type=str)
    parser.add_argument('--latest', '-l', dest='latest', action='store_true')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_true')
    args = parser.parse_args()
    
    config_loader = Config(config_path=args.config)
    outdir = config_loader.base_dir / 'model_weights'
    outdir.mkdir(exist_ok=True, parents=True)
    if args.checkpoint is not None:
        all_weights = [args.checkpoint]
    elif args.latest:
        all_weights = [None]  # default
    else:
        all_weights = [(config_loader.weights_dir / x.stem).as_posix() for x in config_loader.weights_dir.iterdir() if
                       x.suffix == '.index']
    
    if args.verbose:
        print(f'\nWeights list: \n{all_weights}\n')
    for weights in all_weights:
        model = config_loader.load_model(weights)
        model.build_model_weights()
        model.save_weights(outdir / f'{config_loader.data_name}_tts_weights_step{model.step}.hdf5')
    print('Done.')
    print(f'Model weights under {outdir}')