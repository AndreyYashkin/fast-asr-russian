# fast-asr-russian

## Overview

This repository is the implementation code of the paper "Development of a compact speech recognition system for mobile devices for the Russian language".
The pipeline is done with NeMo toolkit.

## Preparing
Install requirements
```
pip3 install -r requirements.txt
```
Download some scripts from NeMo which are not included in the install.
```
python3 utils/update_NeMo_scripts.py
```

## Getting data

```
python3 datasets/get_golos_dataset.py -d data/golos --wav
```
If you are not planing to train, then you can download Golos in opus format instead of wav.
```
python3 datasets/get_commonvoice_data.py --data_root data/mcv
```

Replace "ё" symbol with "е" as it is done in Golos
```
sed -i 's/u0451/u0435/' data/mcv/commonvoice_dev_manifest.json
sed -i 's/u0451/u0435/' data/mcv/commonvoice_test_manifest.json
sed -i 's/u0451/u0435/' data/mcv/commonvoice_train_manifest.json
```

## Training
Create word piece tokenization
```
python process_asr_text_tokenizer.py \
    --manifest=data/golos/train_opus/train_all_golos.jsonl,data/mcv/commonvoice_train_manifest.json \
    --data_root=data/an4 \
    --vocab_size=256 \
    --tokenizer="spe" \
    --spe_type="unigram" \
    --log \
    # --spe_max_sentencepiece_length=???
```
Check that it is posible to compute CTC loss for the most of samples.
```
python3 ctc_loss_check.py --config-name=finetune_citrinet_256_eng
```
Finetune the pretrained english model after you download "STT En Citrinet 256" from NVIDIA NGC and
put in `nemo_experiments/stt_en_citrinet_256.nemo`
```
python3 speech_to_text_finetune.py --config-name=finetune_citrinet_256_eng
```
Train finetuned model after editing `init_from_ptl_ckpt` in `conf/citrinet_256_ru.yaml`
```
python3 speech_to_text_ctc_bpe.py --config-path=conf --config-name=citrinet_256_ru
```

## Getting metrics
To compute metrics for data in `$MANIFEST_PATH`
```
python speech_to_text_eval.py model_path=nemo_experiments/Citrinet-256-8x-Stride-ru/.../checkpoints/Citrinet-256-8x-Stride-ru.nemo dataset_manifest="$MANIFEST_PATH"
```

## Check mobile performance (Android)
To convert model to format that can be used on mobile see notebook in `mobile`.
[Follow pytorch tutorial to measure Android performance](https://pytorch.org/tutorials/recipes/mobile_perf.html).
Make sure that you mobile torch version is built with [fft support](https://github.com/pytorch/pytorch/commit/5045c18bd1214d8bdb3dc41306da9de06868aecb)
```
adb push trace_c_i.ts  /data/local/tmp
adb shell "/data/local/tmp/speed_benchmark_torch --model=/data/local/tmp/trace_c_i.ptl" --no_inputs true --iter 25
```
