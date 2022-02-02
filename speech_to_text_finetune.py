import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)


@hydra_runner(config_path="conf", config_name="config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    asr_model = EncDecCTCModelBPE(cfg=cfg.model, trainer=trainer)

    init_model = EncDecCTCModelBPE.restore_from(cfg.init_from_model, map_location=asr_model.device)
    enc_state_dict = init_model.encoder.state_dict()
    dec_state_dict = init_model.decoder.state_dict()

    asr_model.encoder.load_state_dict(enc_state_dict)
    try:
        asr_model.decoder.load_state_dict(dec_state_dict)
        logging.info("Decoder was restored from pre-trained model")
    except:
        logging.info("Decoder CANNOT be restored from pre-trained model")
    del init_model, enc_state_dict, dec_state_dict

    if cfg.freeze_encoder:
        asr_model.encoder.freeze()
        logging.info("Encoder was freezed")
        if cfg.enable_bn_se:
            asr_model.encoder.apply(enable_bn_se)
            logging.info("SqueezeExcite and BatchNorm in encoder were unfreezed")

    trainer.fit(asr_model)

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        gpu = 1 if cfg.trainer.gpus != 0 else 0
        test_trainer = pl.Trainer(
            gpus=gpu,
            precision=trainer.precision,
            amp_level=trainer.accelerator_connector.amp_level,
            amp_backend=cfg.trainer.get("amp_backend", "native"),
        )
        if asr_model.prepare_test(test_trainer):
            test_trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
