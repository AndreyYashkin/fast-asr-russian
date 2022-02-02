import argparse
import subprocess
import sys


commit = '75fd7439b972a169d22a23c6c8cb948278e56ea1' # 1.6.0


def update(nemo, path, location='./'):
    url = nemo + path
    args = ['wget', '--backups=1', '-P', location, url]
    command = ' '.join(args)
    subprocess.run(command, shell=True, stderr=sys.stderr, stdout=sys.stdout, capture_output=False)


def main():
    parser = argparse.ArgumentParser(description='Downloads necessary scripts from NeMo examples')
    parser.add_argument('-c', '--commit', default=commit, type=str, help='From which commit (branch) you want to to download scripts')
    args = parser.parse_args()

    NeMo = 'https://raw.githubusercontent.com/NVIDIA/NeMo/' + args.commit + '/'

    update(NeMo, 'examples/asr/asr_ctc/speech_to_text_ctc_bpe.py') # train
    update(NeMo, 'examples/asr/transcribe_speech.py') # transcribe text
    update(NeMo, 'examples/asr/transcribe_speech_parallel.py') # transcribe text faster
    update(NeMo, 'examples/asr/speech_to_text_eval.py') # compute wer/cer

    update(NeMo, 'scripts/dataset_processing/get_commonvoice_data.py', 'datasets')
    update(NeMo, 'scripts/tokenizers/process_asr_text_tokenizer.py')


if __name__ == '__main__':
    main()
