"""
Train a diffusion model on images.
"""

import argparse
import json
import os
import random
from functools import partial

import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets import Dataset

from omegaconf import OmegaConf as om
from torch import Tensor
from transformers import set_seed

import wandb
from basic_utils import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    load_defaults_config,
    load_model_emb,
    load_tokenizer,
)
from diffuseq.batch import EncoderBatch
from diffuseq.config import TrainingConfig
from diffuseq.step_sample import create_named_schedule_sampler
from diffuseq.text_datasets import ShorctutFmDataset, load_data_text
from diffuseq.utils import dist_util, logger
from train_util import TrainLoop

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_MODE"] = "online"


def create_argparser():
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)  # update latest args according to argparse
    return parser


def parse_config() -> TrainingConfig:
    """Parse and validate training config from YAML file"""
    with open("configs/config.yaml", "r") as f:
        yaml_cfg = om.load(f)

    # Convert to dict and validate with Pydantic
    config_dict = om.to_container(yaml_cfg, resolve=True)
    training_config = TrainingConfig(**config_dict)

    return training_config


def collate(
        batch: list[dict[str, Tensor]],
        mark_first_padding: bool = False,
        mark_second_padding: bool = False
) -> EncoderBatch:
    """Collates a batch of dictionaries into an EncoderBatch.

    Args:
        batch: A list of dictionaries, where each dictionary represents a single
            item in the batch and contains the keys "seqs", "padding_mask", and
            "input_ids_mask" with corresponding tensors.
        mark_first_padding: If True, marks the first padding token (0) as 1 in padding_mask.
        mark_second_padding: If True, marks the second padding token (0) as 1 in padding_mask.

    Returns:
        An EncoderBatch object containing the collated tensors.
    """

    random.shuffle(batch)

    # Transpose the list of dictionaries into a dictionary of lists
    transposed_batch = {k: [item[k] for item in batch] for k in batch[0]}

    # Stack the tensors along the first dimension (batch dimension)
    collated_batch = {k: torch.stack(v) for k, v in transposed_batch.items()}

    # Modify padding_mask if requested
    if mark_first_padding or mark_second_padding:
        padding_mask = collated_batch["padding_mask"]
        batch_size, seq_len = padding_mask.shape

        for i in range(batch_size):
            # Find indices where padding starts (0s)
            padding_indices = (padding_mask[i] == 0).nonzero(as_tuple=True)[0]

            if len(padding_indices) > 0 and mark_first_padding:
                # Mark first padding token as 1
                padding_mask[i, padding_indices[0]] = 1

            if len(padding_indices) > 1 and mark_second_padding:
                # Mark second padding token as 1
                padding_mask[i, padding_indices[1]] = 1

        collated_batch["padding_mask"] = padding_mask

    return EncoderBatch(**collated_batch)


def create_dataloaders(cfg: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from config.

    :param cfg: Training configuration
    :type cfg: TrainingConfig
    :return: Train and validation dataloaders
    :rtype: tuple[DataLoader, DataLoader]
    """
    logger.info("Loading dataset...")
    train_ds = Dataset.load_from_disk(str(cfg.training_data_path))
    train_text_ds = ShorctutFmDataset(train_ds)
    logger.info(f"Train dataset contains {len(train_ds)} samples.")

    val_ds = Dataset.load_from_disk(str(cfg.validation_data_path))
    val_text_ds = ShorctutFmDataset(val_ds)
    logger.info(f"Validation dataset contains {len(val_ds)} samples.")

    configured_collate = partial(
        collate,
        mark_first_padding=cfg.padding_strategy.mark_first_padding,
        mark_second_padding=cfg.padding_strategy.mark_second_padding
    )

    sampler = DistributedSampler(train_text_ds, shuffle=False)
    train_dataloader = DataLoader(
        train_text_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=8,
        persistent_workers=True,
    )

    sampler = DistributedSampler(val_text_ds, shuffle=False)
    val_dataloader = DataLoader(
        val_text_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=8,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader


def main():
    print(f"cuda available: {torch.cuda.is_available()}")
    args = create_argparser().parse_args()
    cfg = parse_config()

    set_seed(args.seed)
    dist_util.setup_dist()
    logger.configure()
    logger.log("### Creating data loader...")

    data, val_data = create_dataloaders(cfg)
    data_valid = None

    print('#' * 30, 'size of vocab', args.vocab_size)

    logger.log("### Creating model and diffusion...")
    # print('#'*30, 'CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    model, diffusion = create_model_and_diffusion(
        cfg=cfg, **args_to_dict(args, load_defaults_config().keys())
    )
    # print('#'*30, 'cuda', dist_util.dev())
    model.to(dist_util.dev())  # DEBUG **
    # model.cuda() #  DEBUG **

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.log(f'### The parameter count is {pytorch_total_params}')
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
    with open(f'{args.checkpoint_path}/training_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if ('LOCAL_RANK' not in os.environ) or (int(os.environ['LOCAL_RANK']) == 0):
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "FMSeq"),
            name=args.checkpoint_path,
        )
        wandb.config.update(args.__dict__, allow_val_change=True)

    logger.log("### Training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        sc_rate=args.sc_rate
    ).run_loop()


if __name__ == "__main__":
    main()
