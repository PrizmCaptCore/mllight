"""LoRA Fine-tuning Trainer"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from .utils import load_config, resolve_config_path, setup_logger


class LoRATrainer:
    """LoRA Fine-tuning Trainer"""

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to YAML config file, config name, or None for default
                Examples:
                    - None: uses default.yaml
                    - "tiny_fast": uses tiny_fast.yaml from package
                    - "/path/to/config.yaml": uses absolute path
        """
        # Resolve config path
        resolved_path = resolve_config_path(config_path)
        self.config_path = Path(resolved_path)
        self.config_dir = self.config_path.parent
        self.config = load_config(str(self.config_path))
        self.logger = setup_logger(
            name=__name__,
            level=self.config.get('misc', {}).get('logging_level', 'info')
        )

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _resolve_path(self, path_str: str, must_exist: bool = False) -> Path:
        """
        Resolve a path string relative to the config file location first,
        then fall back to the current working directory.
        """
        raw_path = Path(path_str)

        if raw_path.is_absolute():
            return raw_path

        candidates = []
        if self.config_dir:
            candidates.append(self.config_dir / raw_path)
        candidates.append(Path.cwd() / raw_path)

        for candidate in candidates:
            if not must_exist or candidate.exists():
                return candidate

        if must_exist:
            checked = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(f"Path not found for '{path_str}'. Checked: {checked}")

        # If nothing existed and we don't require existence, default to config-relative
        return candidates[0]

    def _get_quantization_config(self) -> BitsAndBytesConfig:
        """Create quantization config from YAML"""
        quant_cfg = self.config['quantization']

        # Convert string dtype to torch dtype
        compute_dtype = quant_cfg['bnb_4bit_compute_dtype']
        if compute_dtype == "bfloat16":
            compute_dtype = torch.bfloat16
        elif compute_dtype == "float16":
            compute_dtype = torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=quant_cfg['load_in_4bit'],
            bnb_4bit_quant_type=quant_cfg['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg['bnb_4bit_use_double_quant'],
        )

    def _get_lora_config(self) -> LoraConfig:
        """Create LoRA config from YAML"""
        lora_cfg = self.config['lora']

        return LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['lora_alpha'],
            target_modules=lora_cfg['target_modules'],
            lora_dropout=lora_cfg['lora_dropout'],
            bias=lora_cfg['bias'],
            task_type=lora_cfg['task_type']
        )

    def load_model(self):
        """Load model and tokenizer"""
        model_cfg = self.config['model']
        model_id = model_cfg['model_id']

        self.logger.info(f"Loading model: {model_id}")

        # Convert string dtype to torch dtype
        torch_dtype_str = model_cfg.get('torch_dtype', 'bfloat16')
        torch_dtype = torch.bfloat16 if torch_dtype_str == "bfloat16" else torch.float16

        # Get quantization config
        bnb_config = self._get_quantization_config()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch_dtype,
            trust_remote_code=model_cfg.get('trust_remote_code', True)
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.logger.info("✓ Model loaded")

    def prepare_model_for_lora(self):
        """Apply LoRA to the model"""
        self.logger.info("Preparing LoRA...")

        # Get LoRA config
        lora_config = self._get_lora_config()

        # Prepare model
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())

        self.logger.info(
            f"Trainable: {trainable_params:,} / {all_params:,} "
            f"({100 * trainable_params / all_params:.2f}%)"
        )

    def load_training_data(self) -> Dataset:
        """Load training data from JSON"""
        data_cfg = self.config['data']
        data_path = self._resolve_path(data_cfg['train_data_path'], must_exist=True)

        self.logger.info(f"Loading data: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Apply prompt template
        template = data_cfg['prompt_template']
        formatted_data = []

        for item in raw_data:
            text = template.format(
                player=item.get('player', ''),
                npc=item.get('npc', '')
            )
            formatted_data.append({"text": text})

        dataset = Dataset.from_list(formatted_data)

        self.logger.info(f"✓ Loaded {len(dataset)} samples")

        return dataset

    def setup_trainer(self, dataset: Dataset):
        """Setup the SFTTrainer"""
        train_cfg = self.config['training']
        output_dir = self._resolve_path(train_cfg['output_dir'])

        # Create TrainingArguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=train_cfg['num_train_epochs'],
            per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
            gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
            learning_rate=train_cfg['learning_rate'],
            lr_scheduler_type=train_cfg['lr_scheduler_type'],
            warmup_ratio=train_cfg['warmup_ratio'],
            optim=train_cfg['optim'],
            bf16=train_cfg['bf16'],
            tf32=train_cfg['tf32'],
            logging_steps=train_cfg['logging_steps'],
            logging_dir=str(output_dir / "logs"),
            save_strategy=train_cfg['save_strategy'],
            save_total_limit=train_cfg['save_total_limit'],
            dataloader_num_workers=train_cfg['dataloader_num_workers'],
            group_by_length=train_cfg['group_by_length'],
            report_to="none"
        )

        # Create SFTTrainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        self.logger.info("✓ Trainer ready")

    def train(self):
        """Start training"""
        self.logger.info("="*60)
        self.logger.info("Starting training...")
        self.logger.info("="*60)

        self.trainer.train()

        self.logger.info("✓ Training complete")

    def save_model(self):
        """Save the trained model"""
        output_dir = self._resolve_path(self.config['training']['output_dir'])

        self.logger.info(f"Saving to: {output_dir}")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.logger.info("✓ Model saved")

    def run(self):
        """Run the complete training pipeline"""
        self.load_model()
        self.prepare_model_for_lora()
        dataset = self.load_training_data()
        self.setup_trainer(dataset)
        self.train()
        self.save_model()


def main():
    """Main entry point"""
    from .utils import list_available_configs

    available = ", ".join(list_available_configs())

    parser = argparse.ArgumentParser(
        description="LoRA Fine-tuning",
        epilog=f"Available configs: {available}"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Config file path or name. "
            "Examples: 'tiny_fast', 'default.yaml', '/path/to/config.yaml'. "
            "Default: default.yaml"
        )
    )

    args = parser.parse_args()

    # Run training
    trainer = LoRATrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()
