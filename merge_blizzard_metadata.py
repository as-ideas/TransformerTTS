import pickle
from pathlib import Path

def scan_blizzard(blizzard_folder: str):
    main_dir = Path(blizzard_folder)
    if not (main_dir / 'original_text_dict.pkl').exists():
        text_data1 = main_dir / 'BC2013_segmented_v0_txt1'
        text_data2 = main_dir / 'BC2013_segmented_v0_txt2'
        text_dirs1 = [d for d in text_data1.iterdir() if d.is_dir()]
        text_dirs2 = [d for d in text_data2.iterdir() if d.is_dir()]
        text_dirs = [text_dirs1, text_dirs2]
        text_dict = {}
        print('Reading text')
        for text_dir in text_dirs:
            for directory in text_dir:
                print(f'directory {directory}')
                files = [f for f in directory.iterdir() if f.suffix == '.txt']
                lines = {f.with_suffix('').name: f.open().readline() for f in files}
                text_dict.update(lines)
        pickle.dump(text_dict, open(main_dir / 'original_text_dict.pkl', 'wb'))
    else:
        print('loading existing text_dict')
        text_dict = pickle.load(open(main_dir / 'original_text_dict.pkl', 'rb'))
    return text_dict

if __name__ == '__main__':
    blizzard_folder = Path('/Volumes/data/datasets/blizzard')
    d = scan_blizzard(blizzard_folder)
    key_list = list(d.keys())
    print('metadata head')
    for key in key_list[:5]:
        print(f'{key}: {d[key]}')
    print('metadata tail')
    for key in key_list[-5:]:
        print(f'{key}: {d[key]}')
    new_metadata = [''.join([key, '|', d[key], '\n']) for key in d]
    original_metadata_path = blizzard_folder / 'original_metadata.txt'
    with open(original_metadata_path, 'w+', encoding='utf-8') as file:
        file.writelines(new_metadata)