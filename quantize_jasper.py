from argparse import ArgumentParser
from tqdm import tqdm

import torch

from nemo.collections.asr.models import EncDecCTCModel
from nemo.utils import logging

from nemo.collections.asr.parts.submodules.jasper import MaskedConv1d
import torch.nn as nn

from torch import __version__ as torch_version
from nemo __version__ as nemo_version

tested_torch_version = '1.10.0'
tested_nemo_version = '1.5.0'


# TODO
try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


class Mask(nn.Module):
    def __init__(self, update_masked_length, get_seq_len):
        super().__init__()
        self.update_masked_length = update_masked_length
        self.get_seq_len = get_seq_len


    def forward(self, x, lens):
        x, lens = self.update_masked_length(x, lens)
        return x, lens


class FakeMaskedConv1d(nn.Module):
    def __init__(self, maskedconv1):
        super().__init__()
        self.mask = Mask(maskedconv1.update_masked_length, maskedconv1.get_seq_len)
        self.not_maskedconv1 = maskedconv1
        self.not_maskedconv1.use_mask = False


    def forward(self, x, lens):
        x, lens = self.mask(x, lens)
        return self.not_maskedconv1(x, lens)


    # To get correct forward arguments Jasper has to think that it is MaskedConv1d
    @property
    def __class__(self):
        return MaskedConv1d


prepare_jasper_config_dict = {
    "non_traceable_module_class": [Mask]
}


def detach_non_traceable(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            detach_non_traceable(module)

        if isinstance(module, MaskedConv1d):
            if module.use_mask:
                module = FakeMaskedConv1d(module)
                setattr(model, name, module)


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dont_normalize_text",
        default=False,
        action='store_true',
        help="Turn off trasnscript normalization. Recommended for non-English.",
    )
    parser.add_argument("--qconfig", type=str, default='qnnpack')
    parser.add_argument("--save_to", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if torch_version.startswith(tested_torch_version) and nemo_version == tested_nemo_version:
        # TODO написать лучше
        warn_str = 'This script was tested with Torch {} and NeMo {}, while current versions are {} and {}, respectively.'.format(
                                                        tested_torch_version, tested_nemo_version, torch_version, nemo_version)
        raise ImportWarning(warn_str)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecCTCModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecCTCModel.from_pretrained(model_name=args.asr_model)

    asr_model.preprocessor.featurizer.pad_to = 0
    asr_model.preprocessor.featurizer.dither = 0.0

    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': 16000,
            'manifest_filepath': args.dataset,
            'labels': asr_model.decoder.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': not args.dont_normalize_text,
            'num_workers': args.num_workers,
        }
    )
    if can_gpu:
        asr_model = asr_model.cuda()

    detach_non_traceable(asr_model)
    asr_model.save_to('hack.nemo') # TODO
    asr_model.eval()

    quantize_fx.prepare_fx(asr_model.encoder, qconfig_dict, prepare_custom_config_dict=prepare_jasper_config_dict, inplace=True)
    quantize_fx.prepare_fx(asr_model.decoder, qconfig_dict, inplace=True)

    for test_batch in tqdm(asr_model.test_dataloader()):
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        #with autocast():
        if True: # TODO
            log_probs, encoded_len, greedy_predictions = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
        # TODO
        #del test_batch
        break

    quantize_fx.convert_fx(asr_model.encoder, inplace=True)
    quantize_fx.convert_fx(asr_model.decoder, inplace=True)

    asr_model.save_to(args.save_to)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
