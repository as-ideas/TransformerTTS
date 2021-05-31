import wave
from argparse import ArgumentParser
import pyaudio
from pathlib import Path

def metadata_reader(metadata_path: str, column_sep='|') -> dict:
    text_dict = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split(column_sep)
            filename, text = l_split[0], l_split[-1]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            text = text.replace('\n', '')
            text_dict.update({filename: text})
    return text_dict

def _prepare_file(fname, mode='wb'):
    wavefile = wave.open(fname, mode)
    wavefile.setnchannels(1)
    wavefile.setsampwidth(PA.get_sample_size(pyaudio.paInt16))
    wavefile.setframerate(22050)
    return wavefile


def callback(in_data, frame_count, time_info, status):
    wavefile.writeframes(in_data)
    return in_data, pyaudio.paContinue


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--target_dir', type=str)
    parser.add_argument('-m', '--source_metadata', type=str)
    args = parser.parse_args()
    PA = pyaudio.PyAudio()
    target_dir = Path(args.target_dir)
    target_dir.mkdir(exist_ok=True)
    text_dict = metadata_reader(args.source_metadata)
    
    if (target_dir / 'metadata.csv').exists():
        saved_metadata = metadata_reader(target_dir / 'metadata.csv')
    else:
        saved_metadata = {}
    
    stop = False
    for key in text_dict:
        if key in saved_metadata:
            continue
        repeat = True
        if stop:
            break
        while repeat:
            print(f"READ: {text_dict[key]}")
            start = input('[enter]: start recording | "stop" | "s": skip')
            if start == 'stop':
                stop = input('Stop? [y]/n')
                if stop != 'n':
                    break
            if start == 's':
                repeat = False
                continue

            wavefile = _prepare_file((target_dir/key).with_suffix('.wav').as_posix())

            _stream = PA.open(format=pyaudio.paInt16,
                              channels=1,
                              rate=22050,
                              input=True,
                              frames_per_buffer=1024,
                              stream_callback=callback)
            value = input('enter to stop, r to repeat')
            _stream.stop_stream()
            if value != 'r':
                repeat = False
                saved_metadata[key] = text_dict[key]

    new_metadata = [f'{k}|{v}\n' for k, v in saved_metadata.items()]
    with open(target_dir / 'metadata.csv', 'w+', encoding='utf-8') as file:
        file.writelines(new_metadata)
