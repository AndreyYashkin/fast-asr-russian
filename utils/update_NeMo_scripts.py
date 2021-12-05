import argparse
import subprocess
import sys


commit_name = 'bump version to 1.5.1'
commit = '01419c3492e3cb20698053a8eb861e703e61751f'


def download(nemo, path, location='./'):
    url = nemo + path
    args = ['wget', '--backups=1', '-P', location, url]
    command = ' '.join(args)
    subprocess.run(command, shell=True, stderr=sys.stderr, stdout=sys.stdout, capture_output=False)


def main():
    parser = argparse.ArgumentParser(description='Downloads necessary scripts from NeMo examples')
    parser.add_argument('-c', '--commit', default=commit, type=str, help='From which commit (branch) you want to to download scripts. Default is commit "{}"'.format(commit_name))
    args = parser.parse_args()

    NeMo = 'https://raw.githubusercontent.com/NVIDIA/NeMo/' + args.commit + '/'

    download(NeMo, 'examples/asr/speech_to_text.py')
    download(NeMo, 'examples/asr/speech_to_text_infer.py')
    download(NeMo, 'examples/asr/transcribe_speech.py')
    #download(NeMo, 'examples/asr/transcribe_speech_parallel.py') # FIXME this one is faster, but is not avaliable yet in 1.5.1

    download(NeMo, 'scripts/dataset_processing/get_commonvoice_data.py', 'datasets')
    download(NeMo, 'scripts/tokenizers/process_asr_text_tokenizer.py')


if __name__ == '__main__':
    main()
