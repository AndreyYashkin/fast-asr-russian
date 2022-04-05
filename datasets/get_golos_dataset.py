import argparse
import json
import os
import tarfile
import subprocess
import sys
from tempfile import TemporaryDirectory


def init(data_root, temp_dir, force, wav):
    if not os.path.isdir(data_root) or force:
        #os.makedirs(data_root, exist_ok = True)
        #download_golos(data_root, temp_dir, wav)
        update_manifests(data_root, init=True)
    else:
        print('Data dir already exits. Will try to update manifests only')
        update_manifests(data_root)


def download(url, dest):
    args = ['wget', '--backups=0',  url, '-O', dest]
    command = ' '.join(args)
    subprocess.run(command, shell=True, stderr=sys.stderr, stdout=sys.stdout, capture_output=False)


def download_golos(data_root, temp_dir, wav):
    if not wav:
        golos_opus_url = 'https://sc.link/JpD'
        golos_tar = 'golos_opus.tar'
        print('Downloading ' + golos_tar)
        tar_path = os.path.join(temp_dir, golos_tar)
        download(golos_opus_url, tar_path)
        print('\nExtracting ' + golos_tar)
    else:
        pairs = [
            ('train_farfield.tar', 'https://sc.link/1Z3'),
            ('train_crowd0.tar', 'https://sc.link/Lrg'),
            ('train_crowd1.tar', 'https://sc.link/MvQ'),
            ('train_crowd2.tar', 'https://sc.link/NwL'),
            ('train_crowd3.tar', 'https://sc.link/Oxg'),
            ('train_crowd4.tar', 'https://sc.link/Pyz'),
            ('train_crowd5.tar', 'https://sc.link/Qz7'),
            ('train_crowd6.tar', 'https://sc.link/RAL'),
            ('train_crowd7.tar', 'https://sc.link/VG5'),
            ('train_crowd8.tar', 'https://sc.link/WJW'),
            ('train_crowd9.tar', 'https://sc.link/XKk'),
            ('test.tar', 'https://sc.link/Kqr'),
            ]

        pairs.reverse()
        for tar, url in pairs:
            tar_path = os.path.join(temp_dir, tar)
            print('Downloading ' + tar)
            download(url, tar_path)
            print('\nExtracting ' + tar)
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(data_root)
            os.remove(tar_path)


def update_manifests(data_root, init = False):
    print('Updating manifests')
    data_root_abs = os.path.abspath(data_root)
    dirs_and_manifests = [('train', [('100hours.jsonl', '100hours.jsonl'),
                                          ('10hours.jsonl', '10hours.jsonl'),
                                          ('10min.jsonl', '10min.jsonl'),
                                          ('1hour.jsonl', '1hour.jsonl'),
                                          ('manifest.jsonl', 'train_all_golos.jsonl')]),
                          ('test/crowd', [('manifest.jsonl', 'test_crowd.jsonl')]),
                          ('test/farfield', [('manifest.jsonl', 'test_farfield.jsonl')]),
                          ]# FIXME test_opus -> test, train_opus
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
                d = json.loads(lines[0])
                cut_abs_path_index = d['audio_filepath'].rindex(important_dir)
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
    parser.add_argument('-w', '--wav', action='store_true', help='Download data in .wav format. Use this option, if you want to train fast')
    args = parser.parse_args()

    with TemporaryDirectory(dir=args.temp_dir) as tmpdirname:
        tmpdirname = '/media/storage/yashkin/tmp'
        init(args.data_root, tmpdirname, args.force, args.wav)



if __name__ == '__main__':
    main()
