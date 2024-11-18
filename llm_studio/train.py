import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import gc
import logging
import sys
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_studio.python_configs.base import DefaultConfigProblemBase
from llm_studio.src.loggers import MainLogger
from llm_studio.src.utils.config_utils import (
    load_config_yaml,
    save_config_yaml,
)
from llm_studio.src.utils.data_utils import (
    get_data,
    get_train_dataloader,
    get_train_dataset,
    get_val_dataloader,
    get_val_dataset,
)
from llm_studio.src.utils.exceptions import LLMTrainingException
from llm_studio.src.utils.export_utils import save_prediction_outputs
from llm_studio.src.utils.logging_utils import (
    TqdmToLogger,
    initialize_logging,
)
from llm_studio.src.utils.modeling_utils import (
    check_disk_space,
    get_optimizer,
    get_scheduler,
    get_torch_dtype,
    load_checkpoint,
    run_inference,
    save_checkpoint,
    save_predictions,
)
from llm_studio.src.utils.utils import (
    create_symlinks_in_parent_folder,
    set_seed,
)

logger = logging.getLogger(__name__)


def run_eval(
    cfg: DefaultConfigProblemBase,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    val_df: pd.DataFrame,
    mode: str = "validation",
) -> Tuple:
    """Runs the evaluation loop.

    Args:
        cfg: config object
        model: trained model
        val_dataloader: validation Dataloader
        val_df: validation DataFrame
        mode: validation

    Returns:
        Validation loss
    """
    with torch.no_grad():
        is_training = model.training
        val_data: Dict[str, Any] = run_inference(
            cfg, model, val_dataloader, mode
        )  # type: ignore
        model.train(is_training)

    val_data = val_dataloader.dataset.postprocess_output(  # type: ignore
        cfg=cfg, df=val_df, output=val_data
    )
    val_loss = np.mean(val_data.get("loss", torch.tensor(0)).float().cpu())
    val_metric = np.mean(val_data["metrics"])
    logger.info(f"{mode.capitalize()} {cfg.prediction.metric}: {val_metric:.5f}")

    cfg.logging._logger.log(
        mode,
        "val_loss",
        val_loss,
        step=cfg.environment._curr_step / cfg.environment._step_log_denominator,
    )
    cfg.logging._logger.log(
        mode,
        cfg.prediction.metric,
        val_metric,
        step=cfg.environment._curr_step / cfg.environment._step_log_denominator,
    )

    save_predictions(cfg, val_data, val_dataloader, val_df, mode)

    return val_loss, val_metric


def run_train(
    cfg: DefaultConfigProblemBase,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch_steps,
    train_dataloader,
    train_df,
    val_dataloader,
    val_df: pd.DataFrame,
):
    """Runs the training loop.

    Args:
        cfg: DefaultConfigProblemBase config object
        model: model
        train_dataloader: custom training Dataloader
        train_df: train DataFrame
        val_dataloader: custom validation Dataloader
        val_df: validation DataFrame

    Returns:
        Validation prediction output
        Validation loss
        Validation metric
        Last train batch
    """

    scaler: GradScaler | None = None
    if cfg.environment.mixed_precision:
        scaler = GradScaler(
            enabled=(cfg.environment.mixed_precision_dtype == "float16")
        )

    optimizer.zero_grad(set_to_none=True)

    # Prepare NLP Augmentation
    nlp_augment = None
    if hasattr(cfg.augmentation, "nlp_augmentations_class"):
        nlp_augment = cfg.augmentation.nlp_augmentations_class(cfg=cfg)

    _, metric_mode, _ = cfg.prediction.metric_class.get(cfg.prediction.metric)
    objective_op: Callable[[float, float], bool]
    if metric_mode == "max":
        best_val_metric = -np.inf
        objective_op = np.greater
    else:
        best_val_metric = np.inf
        objective_op = np.less

    for epoch in range(cfg.training.epochs + 1):
        set_seed(
            cfg.environment._seed
            + epoch * cfg.environment._world_size * cfg.environment.number_of_workers
            + cfg.environment._local_rank * cfg.environment.number_of_workers
        )
        logger.info(f"Training Epoch: {epoch + 1} / {cfg.training.epochs}")

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        progress_bar = tqdm(
            total=epoch_steps,
            disable=cfg.environment._local_rank != 0,
            file=tqdm_out,
            ascii=True,
            desc="train loss",
            mininterval=0,
        )
        tr_it = iter(train_dataloader)

        losses = []
        model.train()

        log_update_steps = max(epoch_steps // 20, 1)
        evaluation_step = max(int(epoch_steps * cfg.training.evaluation_epochs), 1)
        logger.info(f"Evaluation step: {evaluation_step}")

        for itr, batch in enumerate(tr_it):
            cfg.environment._curr_step += (
                cfg.training.batch_size * cfg.environment._world_size
            )

            # NLP augmentation
            if nlp_augment is not None:
                batch = nlp_augment(batch)

            # Forward pass
            with autocast(
                enabled=cfg.environment.mixed_precision,
                dtype=get_torch_dtype(cfg.environment.mixed_precision_dtype),
            ):
                output_dict = model.forward(batch)

            loss = output_dict["loss"]
            if ~np.isfinite(loss.item()) and (itr > 20):
                raise LLMTrainingException(
                    "NaN caught in loss during training. "
                    "Please, reduce learning rate, change dtype, "
                    "or disable mixed precision. Alternatively, "
                    "gradient clipping may help to stabilize training."
                )
            losses.append(loss.item())

            # Backward pass
            if (
                cfg.environment.mixed_precision
                and len(cfg.environment.gpus)
            ):
                if itr % cfg.training.grad_accumulation == 0:
                    scaler.step(optimizer)  # type: ignore
                    scaler.update()
                scaler.scale(loss).backward()  # type: ignore
            else:
                if itr % cfg.training.grad_accumulation == 0:
                    optimizer.step()
                loss.backward()

            if shedule is not None:
                shedule.step()

            if cfg.environment._local_rank == 0:
                cfg.logging._logger.log(
                    "train",
                    "loss",
                    losses[-1],
                    step=cfg.environment._curr_step
                    / cfg.environment._step_log_denominator,
                )
                cfg.logging._logger.log(
                    "meta",
                    "lr",
                    optimizer.param_groups[0]["lr"],
                    step=cfg.environment._curr_step
                    / cfg.environment._step_log_denominator,
                )
                if cfg.training.differential_learning_rate_layers:
                    cfg.logging._logger.log(
                        "meta",
                        "lr_diff",
                        optimizer.param_groups[2]["lr"],
                        step=cfg.environment._curr_step
                        / cfg.environment._step_log_denominator,
                    )

                cfg.logging._logger.log(
                    "internal",
                    "current_step",
                    cfg.environment._curr_step,
                )

                # Show logs each 5% of the epoch (only if doing per epoch evaluation)
                if (itr + 1) % log_update_steps == 0 or itr == epoch_steps - 1:
                    progress_bar.set_description(
                        f"train loss: {np.mean(losses[-10:]):.2f}", refresh=False
                    )
                    if (itr + 1) % log_update_steps == 0:
                        progress_bar.update(log_update_steps)
                    else:
                        progress_bar.update(epoch_steps % log_update_steps)

                del output_dict

            # Validation loop
            if (itr + 1) % evaluation_step == 0:
                if cfg.training.save_checkpoint == "last":
                    logger.info(
                        f"Saving last model checkpoint to {cfg.output_directory}"
                    )
                    save_checkpoint(model=model, path=cfg.output_directory, cfg=cfg)
                elif cfg.training.save_checkpoint == "each_evaluation_epoch":
                    checkpoint_path = os.path.join(
                        cfg.output_directory, f"epoch_{epoch}_step_{itr}"
                    )
                    logger.info(f"Saving model checkpoint to {checkpoint_path}")
                    save_checkpoint(model=model, path=checkpoint_path, cfg=cfg)
                    create_symlinks_in_parent_folder(checkpoint_path)

                val_loss, val_metric = run_eval(
                    cfg, model, train_dataloader, train_df
                )

                if cfg.training.save_checkpoint == "best":
                    if objective_op(val_metric, best_val_metric):
                        logger.info(
                            f"Saving best model checkpoint: "
                            f"val_{cfg.prediction.metric} {best_val_metric:.5} -> "
                            f"{val_metric:.5} to {cfg.output_directory}"
                        )
                        save_checkpoint(model=model, path=cfg.output_directory, cfg=cfg)
                        best_val_metric = val_metric

                model.train()

        progress_bar.close()
        del progress_bar

        cfg.logging._logger.log("internal", "epoch", epoch + 1)

    return val_loss, val_metric


def run(cfg: DefaultConfigProblemBase) -> float:
    """Runs the routine.

    Args:
        cfg: DefaultConfigProblemBase config object with all the hyperparameters
    """

    os.makedirs(cfg.output_directory, exist_ok=True)

    # Set the random seed for reproducibility
    # either random seed when user set it -1 or deterministic user chosen seed
    if cfg.environment.seed < 0:
        cfg.environment._seed = np.random.randint(1_000_000)
    else:
        cfg.environment._seed = cfg.environment.seed

    # Prepare environment
    cfg.environment._distributed = False
    cfg.environment._local_rank = 0

    initialize_logging(cfg)

    cfg.environment._device = (
        "cuda:0"
        if (torch.cuda.is_available() and len(cfg.environment.gpus) > 0)
        else "cpu"
    )
    if cfg.environment._device == "cpu":
        logger.warning("Training on CPU. This will be slow.")

    set_seed(cfg.environment._seed)
    logger.info(f"Problem Type: {cfg.problem_type}")
    logger.info(f"Global random seed: {cfg.environment._seed}")

    # we need to get train dataframe and number of labels if not set or in training mode
    logger.info("Preparing the data...")
    train_df, val_df = get_data(cfg)

    if (
        len(val_df) > int(os.getenv("GPT_EVAL_MAX", 100))
        and "GPT" in cfg.prediction.metric
    ):
        logger.warning(
            f"More than {os.getenv('GPT_EVAL_MAX', 100)} validation records. "
            "Safeguarding against OpenAI API costs. Setting metric to BLEU. "
            "Change GPT_EVAL_MAX to run GPT validation."
        )
        cfg.prediction.metric = "BLEU"

    # prepare data
    logger.info("Preparing train and validation data")
    train_dataset = get_train_dataset(train_df=train_df, cfg=cfg)
    val_dataset = get_val_dataset(val_df=val_df, cfg=cfg)
    train_dataloader = get_train_dataloader(train_ds=train_dataset, cfg=cfg)
    val_dataloader = get_val_dataloader(val_ds=val_dataset, cfg=cfg)
    train_dataloader.shuffle = False
    val_dataloader.shuffle = True

    # Prepare model and optimizer
    model = cfg.architecture.model_class(cfg)
    check_disk_space(model, cfg.output_directory)

    # load model weights
    if cfg.architecture.pretrained_weights != "":
        # Do not load strictly if continue training from the previous experiment
        load_checkpoint(cfg, model, strict=cfg.training.epochs == -1)
    model.to(cfg.environment._device)

    epoch_steps = len(train_dataloader)
    optimizer = get_optimizer(model=model, cfg=cfg)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer, epoch_steps=epoch_steps)

    if cfg.environment.compile_model:
        model.backbone = torch.compile(model.backbone)

    # reset steps
    cfg.environment._curr_step = 0
    cfg.environment._curr_val_step = 0

    gc.collect()

    cfg.logging._logger = MainLogger(cfg)

    _, val_metric = run_train(
        cfg,
        model,
        scheduler,
        optimizer,
        epoch_steps,
        train_dataloader,
        train_df,
        val_dataloader,
        val_df,
    )

    experiment_path = f"{cfg.output_directory}"
    save_config_yaml(f"{cfg.output_directory}/cfg.yaml", cfg)
    save_prediction_outputs(cfg.experiment_name, experiment_path)

    return val_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-Y", "--yaml", help="yaml filename", type=(str), default=argparse.SUPPRESS
    )
    parser_args, unknown = parser.parse_known_args(sys.argv)

    if "yaml" in parser_args:
        cfg = load_config_yaml(parser_args.yaml)
    else:
        raise ValueError("Please, provide a configuration file")

    out_dir = cfg.output_directory
    os.makedirs(out_dir, exist_ok=True)

    try:
        run(cfg=cfg)
    except Exception:
        logging.error("Exception occurred during the run:", exc_info=True)
