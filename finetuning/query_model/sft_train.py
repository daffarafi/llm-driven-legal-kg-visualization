"""
SFT Training Entry Point for NL → Cypher Query Model.

Follows the gllm_training reference pattern:
1. Load YAML config (experiment_config.yml)
2. Initialize SFTTrainer with config
3. Train with LoRA on Qwen3-4b

Run on cloud GPU:
    python finetuning/query_model/sft_train.py --experiment experiment_1

Or with CSV data (no Google Sheets needed):
    python finetuning/query_model/sft_train.py --experiment experiment_2
"""

import logging
import argparse
import sys
import os

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NL → Cypher query model")
    parser.add_argument(
        "--experiment",
        default="experiment_2",
        help="Experiment name from experiment_config.yml (default: experiment_2 = CSV mode)",
    )
    parser.add_argument(
        "--config",
        default="finetuning/query_model/experiment_config.yml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    # Try to use gllm_training library
    try:
        from gllm_training.config.training_config_loader import YamlConfigLoader
        from gllm_training.sft_trainer.sft_trainer import SFTTrainer

        logger.info(f"Loading config: {args.config} → {args.experiment}")

        base_dir = os.path.dirname(args.config)
        config_file = os.path.basename(args.config)

        config_loader = YamlConfigLoader(base_dir=base_dir)
        config = config_loader.load(config_file, args.experiment)

        logger.info(f"Model: {config.get('model_name', 'N/A')}")
        logger.info(f"LoRA rank: {config.get('r', 'N/A')}")
        logger.info(f"Epochs: {config.get('num_train_epochs', 'N/A')}")

        finetuner = SFTTrainer(**config)
        results = finetuner.train()

        logger.info(f"Training complete! Results: {results}")

    except ImportError:
        logger.warning("gllm_training not installed. Using standalone Unsloth training.")
        _train_standalone(args)


def _train_standalone(args):
    """Fallback: standalone training without gllm_training library."""
    import yaml

    with open(args.config, "r") as f:
        full_config = yaml.safe_load(f)

    config = full_config.get(args.experiment, {})
    hyperparams = config.get("hyperparameters", config)

    model_name = config.get("model_name", "Qwen/Qwen3-4b")
    max_seq_length = hyperparams.get("max_seq_length", 4096)
    load_in_4bit = hyperparams.get("load_in_4bit", False)
    load_in_8bit = hyperparams.get("load_in_8bit", True)

    logger.info(f"Standalone training: {model_name}")

    # Load model
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=hyperparams.get("r", 16),
        lora_alpha=hyperparams.get("lora_alpha", 16),
        target_modules=hyperparams.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=hyperparams.get("lora_dropout", 0.1),
        bias=hyperparams.get("bias", "none"),
        use_gradient_checkpointing=hyperparams.get("use_gradient_checkpointing", "unsloth"),
        random_state=hyperparams.get("random_state", 3407),
    )

    # Load dataset
    from datasets import load_dataset

    datasets_path = config.get("datasets_path", "finetuning/query_model/data")
    train_file = os.path.join(datasets_path, config.get("train_filename", "training_data.csv"))
    val_file = os.path.join(datasets_path, config.get("validation_filename", "validation_data.csv"))

    dataset = load_dataset("csv", data_files={"train": train_file, "validation": val_file})

    # Format prompt
    def format_prompt(sample):
        return (
            f"### Instruction:\n{sample['context']}\n\n"
            f"### Input:\n{sample['question']}\n\n"
            f"### Output:\n{sample['response']}"
        )

    # Train
    from trl import SFTTrainer as TRLSFTTrainer, SFTConfig

    output_dir = hyperparams.get("model_output_dir", "data/fine_tuned")

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=hyperparams.get("num_train_epochs", 4),
        per_device_train_batch_size=hyperparams.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
        learning_rate=hyperparams.get("learning_rate", 2e-5),
        warmup_steps=hyperparams.get("warmup_steps", 20),
        logging_steps=hyperparams.get("logging_steps", 10),
        save_strategy=hyperparams.get("save_strategy", "steps"),
        save_steps=hyperparams.get("save_steps", 25),
        save_total_limit=hyperparams.get("save_total_limit", 3),
        eval_strategy=hyperparams.get("eval_strategy", "steps"),
        eval_steps=hyperparams.get("eval_steps", 25),
        load_best_model_at_end=hyperparams.get("load_best_model_at_end", True),
        metric_for_best_model=hyperparams.get("metric_for_best_model", "eval_loss"),
        greater_is_better=hyperparams.get("greater_is_better", False),
        optim=hyperparams.get("optim", "adamw_8bit"),
        weight_decay=hyperparams.get("weight_decay", 0.1),
        lr_scheduler_type=hyperparams.get("lr_scheduler_type", "linear"),
        seed=hyperparams.get("seed", 3407),
        report_to=hyperparams.get("report_to", "none"),
    )

    trainer = TRLSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        formatting_func=format_prompt,
        args=training_args,
    )

    trainer.train()

    # Save
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
