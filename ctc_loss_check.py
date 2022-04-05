import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def analyse_ctc_failures_in_model(model):
    count_ctc_failures = 0
    am_seq_lengths = []
    target_seq_lengths = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    mode = model.training

    train_dl = model.train_dataloader()

    with torch.no_grad():
      model = model.eval()
      for batch in tqdm(train_dl, desc='Checking for CTC failures'):
          x, x_len, y, y_len = batch
          x, x_len = x.to(device), x_len.to(device)
          x_logprobs, x_len, greedy_predictions = model(input_signal=x, input_signal_length=x_len)

          # Find how many CTC loss computation failures will occur
          for xl, yl in zip(x_len, y_len):
              if xl <= yl:
                  count_ctc_failures += 1

          # Record acoustic model lengths=
          am_seq_lengths.extend(x_len.to('cpu').numpy().tolist())

          # Record target sequence lengths
          target_seq_lengths.extend(y_len.to('cpu').numpy().tolist())

          del x, x_len, y, y_len, x_logprobs, greedy_predictions

    if mode:
      model = model.train()

    return count_ctc_failures, am_seq_lengths, target_seq_lengths


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    #asr_model = EncDecCTCModelBPE.restore_from(cfg.init_from_model)
    asr_model = EncDecCTCModelBPE(cfg=cfg.model)
    num_ctc_failures, am_seq_lengths, target_seq_lengths = analyse_ctc_failures_in_model(asr_model)

    if num_ctc_failures > 0:
        logging.warning(f"\nCTC loss will fail for {num_ctc_failures} samples ({num_ctc_failures * 100./ float(len(am_seq_lengths))} % of samples)!\n"
                        f"Increase the vocabulary size of the tokenizer so that this number becomes close to zero !")
    else:
        logging.info("No CTC failure cases !")

    # Compute average ratio of T / U
    avg_T = sum(am_seq_lengths) / float(len(am_seq_lengths))
    avg_U = sum(target_seq_lengths) / float(len(target_seq_lengths))

    avg_length_ratio = 0
    for am_len, tgt_len in zip(am_seq_lengths, target_seq_lengths):
        if float(tgt_len) != 0:
            avg_length_ratio += (am_len / float(tgt_len))
        avg_length_ratio = avg_length_ratio / len(am_seq_lengths)

    print(f"Average Acoustic model sequence length = {avg_T}")
    print(f"Average Target sequence length = {avg_U}")
    print()
    print(f"Ratio of Average AM sequence length to target sequence length = {avg_length_ratio}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
