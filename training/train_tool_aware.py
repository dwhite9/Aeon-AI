"""
QLoRA Fine-tuning for Tool-Aware Mistral 7B

Memory-optimized training script for RTX 4070 Ti (12GB VRAM).
Uses 4-bit quantization, gradient checkpointing, and paged optimizers.
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import wandb

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration optimized for 12GB VRAM"""

    # Model settings
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    output_dir: str = "./output"

    # QLoRA settings (4-bit quantization)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # Use bfloat16 for training
    bnb_4bit_quant_type: str = "nf4"  # Normal Float 4-bit
    bnb_4bit_use_double_quant: bool = True  # Nested quantization for memory

    # LoRA settings
    lora_r: int = 8  # Rank (8 is good for 12GB, can try 16 if memory allows)
    lora_alpha: int = 16  # Scaling factor (typically 2*r)
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # FFN
    ])

    # Training hyperparameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 1  # Small batch for 12GB
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Memory optimization
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"  # Paged optimizer saves memory
    max_seq_length: int = 2048  # Shorter than model max to save memory

    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 2  # Keep only 2 checkpoints
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Data paths
    train_data_path: str = "./data/train.json"
    val_data_path: str = "./data/val.json"

    # Weights & Biases (optional)
    use_wandb: bool = False
    wandb_project: str = "mistral-tool-aware"
    wandb_run_name: Optional[str] = None

    # HuggingFace token
    hf_token: Optional[str] = None


def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(
                f"GPU {i}: {mem_allocated:.2f}GB allocated, "
                f"{mem_reserved:.2f}GB reserved, "
                f"{mem_total:.2f}GB total"
            )


def load_model_and_tokenizer(config: TrainingConfig):
    """
    Load Mistral model with 4-bit quantization

    Args:
        config: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {config.model_name}")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=config.hf_token,
        trust_remote_code=True
    )

    # Set pad token (Mistral doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically distribute across available GPUs
        token=config.hf_token,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Prepare model for k-bit training (adds hooks for gradient tracking)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Add LoRA adapters
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"Total params: {total_params:,} || "
        f"Trainable %: {100 * trainable_params / total_params:.2f}%"
    )

    print_gpu_memory()

    return model, tokenizer


def format_instruction_mistral(instruction: str, response: str) -> str:
    """
    Format training example in Mistral instruction format

    Mistral uses: <s>[INST] {instruction} [/INST] {response}</s>
    """
    return f"<s>[INST] {instruction} [/INST] {response}</s>"


def load_and_prepare_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int
) -> Dataset:
    """
    Load and tokenize dataset

    Args:
        data_path: Path to JSON data file
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length

    Returns:
        Tokenized dataset
    """
    logger.info(f"Loading dataset from: {data_path}")

    # Load JSON data
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Format as Mistral instructions
    formatted_texts = []
    for item in data:
        text = format_instruction_mistral(
            item['instruction'],
            item['response']
        )
        formatted_texts.append(text)

    # Create dataset
    dataset = Dataset.from_dict({"text": formatted_texts})

    # Tokenize
    def tokenize_function(examples):
        # Tokenize with truncation and padding
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )

        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()

        # Set pad tokens to -100 so they're ignored in loss
        tokenized["labels"][tokenized["labels"] == tokenizer.pad_token_id] = -100

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    logger.info(f"Dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset


def train(config: TrainingConfig):
    """
    Main training function

    Args:
        config: Training configuration
    """
    logger.info("=" * 60)
    logger.info("Starting Tool-Aware Fine-tuning")
    logger.info("=" * 60)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or f"mistral-tool-aware-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config.__dict__
        )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Load datasets
    train_dataset = load_and_prepare_dataset(
        config.train_data_path,
        tokenizer,
        config.max_seq_length
    )
    eval_dataset = load_and_prepare_dataset(
        config.val_data_path,
        tokenizer,
        config.max_seq_length
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        evaluation_strategy=config.evaluation_strategy,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,  # For loss
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=False,  # Don't use fp16 with bfloat16
        bf16=True,  # Use bfloat16
        report_to="wandb" if config.use_wandb else "none",
        save_safetensors=True,
        dataloader_num_workers=0,  # Prevent multiprocessing issues
        ddp_find_unused_parameters=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Print memory before training
    logger.info("\nMemory usage before training:")
    print_gpu_memory()

    # Train
    logger.info("\nStarting training...")
    trainer.train()

    # Save final model
    logger.info("\nSaving final model...")
    final_output_dir = output_dir / "final_model"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))

    # Save LoRA adapters separately (these are small)
    model.save_pretrained(str(output_dir / "lora_adapters"))

    logger.info("\nMemory usage after training:")
    print_gpu_memory()

    logger.info("\n" + "=" * 60)
    logger.info("✓ Training complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info(f"LoRA adapters saved to: {output_dir / 'lora_adapters'}")

    if config.use_wandb:
        wandb.finish()


def main():
    """Main entry point"""
    # Load config from environment or use defaults
    config = TrainingConfig(
        model_name=os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2"),
        output_dir=os.getenv("OUTPUT_DIR", "./output"),
        train_data_path=os.getenv("TRAIN_DATA", "./data/train.json"),
        val_data_path=os.getenv("VAL_DATA", "./data/val.json"),
        hf_token=os.getenv("HF_TOKEN"),
        use_wandb=os.getenv("USE_WANDB", "false").lower() == "true",
        wandb_project=os.getenv("WANDB_PROJECT", "mistral-tool-aware"),
    )

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available! This script requires a GPU.")
        return

    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA Version: {torch.version.cuda}")

    # Run training
    train(config)


if __name__ == "__main__":
    main()
