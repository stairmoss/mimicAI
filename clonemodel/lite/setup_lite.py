import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_SHARD_DIR = os.path.expanduser("~/.clonemodel_lite/shards")


def run_setup(model_name="k2-fsa/OmniVoice", output_dir=DEFAULT_SHARD_DIR, bits=4, group_size=128, skip_tokenizer=False):
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"

    if manifest_path.exists():
        logger.info(f"Shards already exist at {output_path}")
        return True

    logger.info("=" * 60)
    logger.info("OmniVoice Lite — First-Time Setup")
    logger.info(f"Model: {model_name} | Output: {output_path} | INT{bits}")
    logger.info("This may take 10-30 minutes depending on internet speed.")
    logger.info("=" * 60)

    from clonemodel.lite.shard_model import shard_model

    try:
        shard_model(model_name=model_name, output_dir=output_dir, bits=bits, group_size=group_size, shard_tokenizer=not skip_tokenizer)
        logger.info("Setup complete!")
        return True
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="OmniVoice Lite first-time setup")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--output", default=DEFAULT_SHARD_DIR)
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    success = run_setup(model_name=args.model, output_dir=args.output, bits=args.bits, group_size=args.group_size, skip_tokenizer=args.skip_tokenizer)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
