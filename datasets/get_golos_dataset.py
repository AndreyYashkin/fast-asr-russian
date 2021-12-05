import argparse
import json
import os
import tarfile
import wget # TODO заменить
from tempfile import TemporaryDirectory


def init(data_root, temp_dir, force):
    if not os.path.isdir(data_root) or force:
        os.makedirs(data_root, exist_ok = True)
        golos_opus_url = 'https://sc.link/JpD'
        golos_tar = 'golos_opus.tar'
        print('Downloading ' + golos_tar)
        tar_path = os.path.join(temp_dir, golos_tar)
        wget.download(golos_opus_url, tar_path)
        print('\nExtracting ' + golos_tar)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(data_root)
        update_manifests(data_root, init=True)
    else:
        print('Data dir already exits. Will try to update manifests only')
        update_manifests(data_root)


def update_manifests(data_root, init = False):
    print('Updating manifests')
    data_root_abs = os.path.abspath(data_root)
    dirs_and_manifests = [('train_opus', [('100hours.jsonl', '100hours.jsonl'),
                                          ('10hours.jsonl', '10hours.jsonl'),
                                          ('10min.jsonl', '10min.jsonl'),
                                          ('1hour.jsonl', '1hour.jsonl'),
                                          ('manifest.jsonl', 'train_all_golos.jsonl')]),
                          ('test_opus/crowd', [('manifest.jsonl', 'test_crowd.jsonl')]),
                          ('test_opus/farfield', [('manifest.jsonl', 'test_farfield.jsonl')]),
                          ]
    for (important_dir, manifests) in dirs_and_manifests:
        for (orig_manifest, new_manifest) in manifests:
            if not init:
                orig_manifest = new_manifest
            manifest_path = os.path.join(data_root, important_dir, orig_manifest)
            with open(manifest_path, 'r') as manifest_file:
                lines = list(manifest_file)
            if init:
                cut_abs_path_index = 0
            else:
                cut_abs_path_index = lines[0]['audio_filepath'].rindex(important_dir)
            new_lines = list()
            for line in lines:
                line_dict = json.loads(line)
                audio_filepath = line_dict['audio_filepath'][cut_abs_path_index:]
                if init:
                    audio_filepath = os.path.join(important_dir, audio_filepath)
                # update abs path to audio files
                audio_filepath = os.path.join(data_root_abs, audio_filepath)
                line_dict['audio_filepath'] = audio_filepath
                new_lines.append(json.dumps(line_dict))
            os.remove(manifest_path)
            manifest_path = os.path.join(data_root, important_dir, new_manifest)
            with open(manifest_path, 'w+') as manifest_file:
                for line in new_lines:
                    manifest_file.write(line + os.linesep)


def main():
    parser = argparse.ArgumentParser(description='Downloads and processes Golos dataset', epilog='This is going to take a while ...')
    parser.add_argument('-d', '--data_root', default='golos', type=str, help='Directory to store the dataset.')
    parser.add_argument('-t','--temp_dir', default=None, type=str, help='Where temp directory shall be created')
    parser.add_argument('-f', '--force', action='store_true', help='Do not check that dataset already exits')
    args = parser.parse_args()

    with TemporaryDirectory(dir=args.temp_dir) as tmpdirname:
        tmpdirname = '/media/storage1/yashkin/1'
        init(args.data_root, tmpdirname, args.force)



if __name__ == '__main__':
    main()
