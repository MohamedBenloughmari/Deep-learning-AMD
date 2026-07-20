import sys
from typing import Optional

from omegaconf import DictConfig, OmegaConf

from amd_oct.utils.logging import get_logger

log = get_logger("amd_oct.cli")

COMMANDS = ("train", "evaluate", "predict", "prepare-data")

USAGE = """\
amd-oct — multimodal AMD/OCT classification (MICCAI Task 2)

Usage:
  amd-oct <command> [hydra overrides...]

Commands:
  train          Train a model from a config.
  evaluate       Evaluate a checkpoint on a split.
  predict        Generate predictions / submission CSV for a split.
  prepare-data   Download / extract / sample the dataset and build CSV manifests.

Examples:
  amd-oct train --config-name=efficientnet_v2_s_best
  amd-oct train --config-name=efficientnet_v2_s_best loss.name=corn training.epochs=20
  amd-oct evaluate --config-name=efficientnet_v2_s_best checkpoint=outputs/run/best.pth
  amd-oct predict  --config-name=efficientnet_v2_s_best checkpoint=outputs/run/best.pth split=test
  amd-oct prepare-data source=kaggle out_dir=data/

Run name and output dir:
  run_name=effnet_corn  output_dir=outputs/effnet_corn
"""


_CURRENT_COMMAND: Optional[str] = None


def _dispatch(cfg: DictConfig) -> None:
    cmd = _CURRENT_COMMAND
    OmegaConf.resolve(cfg)
    log.info(f"Command: {cmd}")
    if cmd == "train":
        from amd_oct.train import train

        train(cfg)
    elif cmd == "evaluate":
        from amd_oct.evaluate import evaluate_cli

        evaluate_cli(cfg)
    elif cmd == "predict":
        from amd_oct.predict import predict_cli

        predict_cli(cfg)
    elif cmd == "prepare-data":
        from amd_oct.data.prepare import prepare_data

        prepare_data(cfg.get("prepare", cfg))
    else:
        raise SystemExit(f"Unknown command '{cmd}'. Expected one of {COMMANDS}.")


def _app() -> None:
    import hydra

    @hydra.main(version_base=None, config_path="configs", config_name="default")
    def _main(cfg: DictConfig) -> None:
        _dispatch(cfg)

    _main()


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return
    command = args[0]
    if command.startswith("-"):
        print(USAGE)
        return
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command '{command}'. Expected one of {COMMANDS}.\n\n{USAGE}")
    global _CURRENT_COMMAND
    _CURRENT_COMMAND = command
    sys.argv = [sys.argv[0]] + args[1:]
    _app()


if __name__ == "__main__":
    main()
