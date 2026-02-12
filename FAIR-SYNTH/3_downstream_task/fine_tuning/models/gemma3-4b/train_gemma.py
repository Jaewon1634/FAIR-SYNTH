import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from utils.common import ModelConfig, SystemMonitor, console
from utils.trainer import BaseTrainer
import argparse

def create_gemma_config():
    """Create Gemma model configuration"""
    return ModelConfig(
        model_name="gemma3-4b",
        model_id="google/gemma-3-4b-it",
        max_length=4096,
        batch_size=2,
        learning_rate=1e-4,
        num_epochs=3,
        warmup_steps=300,
        save_steps=500,
        eval_steps=250,
        gradient_accumulation_steps=8,
        fp16=False,
        bf16=True,
        use_lora=False,
        use_qlora=False,
        gradient_checkpointing=True,
        logging_steps=10,
        logging_first_step=True,
        objective="causal_lm"
    )

def main():
    parser = argparse.ArgumentParser(description="Gemma-3-4B-IT QLoRA Fine-tuning")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data (JSONL format)")
    parser.add_argument("--output_dir", type=str, 
                       default="checkpoints/gemma3-4b",
                       help="Model checkpoint output directory")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA fine-tuning")
    parser.add_argument("--use_qlora", action="store_true",
                        help="Use QLoRA (4-bit quantization)")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Override max sequence length")
    
    args = parser.parse_args()
    
    SystemMonitor.print_system_info()
    
    config = create_gemma_config()
    
    # Apply LoRA/QLoRA settings
    if args.use_lora:
        config.use_lora = True
        config.use_qlora = args.use_qlora
        if config.use_qlora:
            config.bf16 = False
            config.fp16 = True
    
    # Override max_length if specified
    if args.max_length is not None:
        config.max_length = args.max_length
    
    # Train
    trainer = BaseTrainer(config, args.data_path, args.output_dir, mode="finetune")
    console.print(f"[cyan]LoRA: {config.use_lora}, QLoRA: {config.use_qlora}[/cyan]")
    
    try:
        trainer.train()
        console.print(f"[bold green]Gemma-3-4B-IT fine-tuning completed![/bold green]")
        console.print(f"[green]Model saved to: {args.output_dir}[/green]")
    except Exception as e:
        console.print(f"[bold red]Training error: {str(e)}[/bold red]")
        raise

if __name__ == "__main__":
    main()
