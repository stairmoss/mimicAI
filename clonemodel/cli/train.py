#!/usr/bin/env python3

"""Training CLI for MimicVoice.

Launches distributed training via HuggingFace Accelerate.
Supports pre-training on Emilia data and finetuning on custom data.

Usage:
    accelerate launch --gpu_ids 0,1,2,3 --num_processes 4 \\
        -m MimicAI.cli.train \\
        --train_config train_config.json \\
        --data_config data_config.json \\
        --output_dir output/

See examples/run_emilia.sh and examples/run_finetune.sh for full pipelines.
"""

import argparse

from clonemodel.training.builder import build_dataloaders, build_model_and_tokenizer
from clonemodel.training.config import TrainingConfig
from clonemodel.training.trainer import OmniTrainer


def main():
    parser = argparse.ArgumentParser(description="MimicVoice Training Entry Point")
    parser.add_argument(
        "--train_config", type=str, required=True, help="Path to config JSON"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save checkpoints"
    )
    parser.add_argument(
        "--data_config", type=str, required=True, help="Path to data config JSON"
    )
    args = parser.parse_args()

    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    model, tokenizer = build_model_and_tokenizer(config)
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
