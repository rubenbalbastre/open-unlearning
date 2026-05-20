import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


def configure_wandb(cfg: DictConfig):
    if os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_DISABLED") != "true":
        os.environ.setdefault("WANDB_PROJECT", "open-unlearning")
        os.environ.setdefault("WANDB_NAME", cfg.task_name)
        with open_dict(cfg):
            cfg.trainer.args.report_to = "wandb"


def apply_lora_to_model(model, model_cfg):
    lora_args = model_cfg.get("lora_args", None)
    if lora_args is not None:
        from peft import LoraConfig, get_peft_model

        num_hidden_layers = model.config.num_hidden_layers
        lora_config = LoraConfig(
            r=lora_args.r,
            lora_alpha=lora_args.alpha,
            init_lora_weights=lora_args.init_strategy,
            target_modules=lora_args.target_modules,
            task_type="CAUSAL_LM",
            bias="none",
            layers_pattern="layers",
            layers_to_transform=list(range(num_hidden_layers - lora_args.number_layers_to_transform, num_hidden_layers))
        )
        model = get_peft_model(model, lora_config)
    return model


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    load_dotenv(Path(get_original_cwd()) / ".env")
    configure_wandb(cfg)
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")

    # Configure Model
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)
    model = apply_lora_to_model(model, model_cfg)
    model.print_trainable_parameters()

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
