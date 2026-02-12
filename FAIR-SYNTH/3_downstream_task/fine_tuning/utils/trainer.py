import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
    EarlyStoppingCallback,
)
try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    prepare_model_for_kbit_training = None
import time
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
import numpy as np
from .common import ModelConfig, TrainingLogger, console, format_time


class ConsoleLoggingCallback(TrainerCallback):
    """Console logging callback for training/evaluation metrics"""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        if not any(key in logs for key in ("loss", "eval_loss")):
            return

        interesting_keys = ("step", "epoch", "loss", "eval_loss", "grad_norm", "learning_rate")
        normalized: Dict[str, Any] = {}
        for key in interesting_keys:
            if key not in logs:
                continue
            value = logs[key]
            if isinstance(value, (np.generic,)):
                value = value.item()
            normalized[key] = value

        if normalized:
            console.print(
                f"[cyan]Log[/cyan] {json.dumps(normalized, ensure_ascii=False)}",
                highlight=False,
            )

class UnsupervisedDataset(Dataset):
    """Dataset class for unsupervised learning (causal language modeling)"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Process texts for natural language learning
        for text in texts:
            if len(text.strip()) < 20:  # Skip very short texts
                continue
                
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
            
            if len(tokens) >= 10:  # Minimum length requirement
                self.examples.append({
                    'input_ids': tokens,
                    'attention_mask': [1] * len(tokens)
                })
        
        console.print(f"[green]Prepared {len(self.examples)} training samples[/green]")
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

class BaseTrainer:
    """Base trainer class for model fine-tuning"""
    
    def __init__(self, config: ModelConfig, data_path: str, output_dir: str, mode: str = "finetune"):
        self.config = config
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        
        # Initialize logger
        self.logger = TrainingLogger(
            log_dir=str(self.output_dir / "logs"),
            model_name=config.model_name
        )
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"[green]Device: {self.device}[/green]")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with QLoRA support"""
        console.print(f"[yellow]Loading model: {self.config.model_id}[/yellow]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # LoRA/QLoRA configuration
        if self.config.use_lora:
            if self.config.use_qlora:
                # QLoRA: 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                # Load model with 4-bit quantization
                model_kwargs = {
                    "quantization_config": bnb_config,
                    "device_map": "auto",
                    "trust_remote_code": True
                }
                
                # For Phi models: use eager attention (without flash-attn)
                if "phi" in self.config.model_id.lower():
                    model_kwargs["attn_implementation"] = "eager"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    **model_kwargs
                )
                
                # Prepare model for QLoRA
                if prepare_model_for_kbit_training is None:
                    raise ImportError("peft is required. Set use_lora=False or install peft.")
                self.model = prepare_model_for_kbit_training(self.model)
                self._apply_lora()
            else:
                # Standard LoRA (no quantization)
                model_kwargs = {
                    "torch_dtype": torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32),
                    "device_map": "auto",
                    "trust_remote_code": True
                }
                
                # For Phi models: use eager attention
                if "phi" in self.config.model_id.lower():
                    model_kwargs["attn_implementation"] = "eager"
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_id,
                    **model_kwargs
                )
                self._apply_lora()
            
        else:
            # Standard mode
            model_kwargs = {
                "torch_dtype": torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32),
                "device_map": "auto",
                "trust_remote_code": True
            }
            
            # For Phi models: use eager attention
            if "phi" in self.config.model_id.lower():
                model_kwargs["attn_implementation"] = "eager"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs
            )
            
        console.print(f"[green]Model loaded successfully[/green]")
        
    def _apply_lora(self):
        """Apply LoRA adapter to model"""
        if LoraConfig is None or get_peft_model is None or TaskType is None:
            raise ImportError("peft is required. Set use_lora=False or install peft.")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
    def load_dataset(self):
        """Load dataset for unsupervised learning"""
        console.print(f"[yellow]Loading dataset: {self.data_path}[/yellow]")
        
        # Read JSONL file
        texts = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'text' in data:
                        texts.append(data['text'])
                    elif 'content' in data:
                        texts.append(data['content'])
                    else:
                        # Try other field names
                        for key in data.keys():
                            if isinstance(data[key], str) and len(data[key]) > 20:
                                texts.append(data[key])
                                break
                except json.JSONDecodeError:
                    continue
        
        if not texts:
            raise ValueError(f"No text found in data file: {self.data_path}")
        
        console.print(f"[green]Loaded {len(texts)} texts[/green]")
        
        # Split dataset (90% train, 10% eval)
        split_idx = int(len(texts) * 0.9)
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:] if len(texts) > 10 else texts[:min(10, len(texts))]
        
        # Create datasets
        self.train_dataset = UnsupervisedDataset(
            train_texts, self.tokenizer, self.config.max_length
        )
        self.eval_dataset = UnsupervisedDataset(
            eval_texts, self.tokenizer, self.config.max_length
        )
        
        console.print(f"[green]Train: {len(self.train_dataset)} samples, Eval: {len(self.eval_dataset)} samples[/green]")
        
    def setup_training_args(self):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=str(self.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            logging_first_step=self.config.logging_first_step,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.config.fp16 and not self.config.bf16,
            bf16=self.config.bf16,
            max_grad_norm=0.1,
            lr_scheduler_type="cosine",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],
        )
    
    def train(self):
        """Execute model training"""
        start_time = time.time()
        
        # Load model and dataset
        self.load_model_and_tokenizer()
        self.load_dataset()
        
        # Setup training
        training_args = self.setup_training_args()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Setup callbacks
        callbacks = [ConsoleLoggingCallback()]
        if self.eval_dataset is not None and len(self.eval_dataset) > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                )
            )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        # Log training start
        config_dict = {
            "model_name": self.config.model_name,
            "model_id": self.config.model_id,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "num_epochs": self.config.num_epochs,
            "use_lora": self.config.use_lora
        }
        self.logger.log_training_start(config_dict)
        
        console.print(f"[green]Training started: {self.config.model_name}[/green]")
        
        # Start training
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(str(self.output_dir))
        
        # Log completion
        total_time = time.time() - start_time
        self.logger.log_training_complete(total_time)
        
        console.print(f"[green]Training completed! Time: {format_time(total_time)}[/green]")
        console.print(f"[green]Model saved to: {self.output_dir}[/green]")
        
        return trainer
