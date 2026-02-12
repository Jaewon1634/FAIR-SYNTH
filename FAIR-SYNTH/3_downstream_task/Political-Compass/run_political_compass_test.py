import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

console = Console()


class PoliticalCompassTester:
    """Political Compass Test for fine-tuned models"""
    
    def __init__(self, checkpoint_path: str = None, base_model_id: str = None):
        """
        Initialize tester
        checkpoint_path: Path to fine-tuned checkpoint (optional, if None uses base model only)
        base_model_id: Base model identifier
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.base_model_id = base_model_id
        self.model = None
        self.tokenizer = None
        
        if self.checkpoint_path and not self.checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    
    def _detect_base_model(self) -> str:
        """Detect base model from checkpoint or use provided base_model_id"""
        if self.base_model_id:
            return self.base_model_id
            
        if not self.checkpoint_path:
            raise ValueError("Either checkpoint_path or base_model_id must be provided")
        
        # Check adapter_config.json
        adapter_config_path = self.checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")
                    if base_model:
                        console.print(f"[cyan]Base model detected: {base_model}[/cyan]")
                        return base_model
            except Exception as e:
                console.print(f"[yellow]Failed to read adapter_config.json: {e}[/yellow]")
        
        # Fallback to path inference
        path_str = str(self.checkpoint_path).lower()
        if "gemma3-4b" in path_str or "gemma" in path_str:
            return "google/gemma-3-4b-it"
        elif "llama3.2-3b" in path_str or "llama" in path_str:
            return "meta-llama/Llama-3.2-3B-Instruct"
        elif "qwen3-4b" in path_str or "qwen" in path_str:
            return "Qwen/Qwen2.5-3B-Instruct"
                
        raise ValueError("Cannot detect base model. Please specify base_model_id")
    
    def load_model(self, use_base_only: bool = False):
        """Load model and tokenizer"""
        base_model_id = self._detect_base_model()
        
        if use_base_only:
            console.print(f"[bold blue]Loading base model only: {base_model_id}[/bold blue]")
        else:
            console.print(f"[bold blue]Loading base model: {base_model_id}[/bold blue]")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            padding_side="left",
            cache_dir="/tmp/huggingface_cache"
        )
        
        # Handle special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 4-bit quantization config
        compute_dtype = torch.bfloat16 if "gemma" in base_model_id.lower() else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": "/tmp/huggingface_cache"
        }
        
        # Model-specific settings
        if "gemma" in base_model_id.lower():
            model_kwargs["attn_implementation"] = "eager"
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif "qwen" in base_model_id.lower():
            model_kwargs["attn_implementation"] = "eager"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            **model_kwargs
        )
        
        # Load LoRA adapter if checkpoint is provided and not using base only
        if self.checkpoint_path and not use_base_only:
            console.print(f"[cyan]Loading LoRA adapter: {self.checkpoint_path.name}[/cyan]")
            self.model = PeftModel.from_pretrained(
                base_model,
                str(self.checkpoint_path),
                is_trainable=False
            )
        else:
            self.model = base_model
        
        self.model.eval()
        
        # Check GPU usage
        if torch.cuda.is_available():
            console.print(f"[green]Model loaded on: {next(self.model.parameters()).device}[/green]")
        
        return True
    
    def generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response for prompt"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generation config
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
            "use_cache": False,
            "output_scores": False,
            "return_dict_in_generate": False,
        }
        
        # Generate response
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
                
                # Extract response
                input_length = inputs["input_ids"].shape[1]
                response_ids = outputs[0][input_length:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
            except Exception as e:
                console.print(f"[red]Generation error: {e}[/red]")
                response = f"Error: {str(e)}"
        
        return response
    
    def ask_question(self, question: str, options: list) -> str:
        """Ask a political compass question"""
        options_text = "\n".join([f"{i+1}. {option}" for i, option in enumerate(options)])
        
        prompt = f"""Please answer the following question by selecting the most appropriate option.

Question: {question}

Options:
{options_text}

Please select the number of your choice and briefly explain your reasoning.
Answer: """
        
        response = self.generate_response(prompt, max_new_tokens=150)
        return response
    
    def load_questions(self, questions_file: str) -> list:
        """Load questions from JSON file"""
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data["questions"]
    
    def run_single_test(self, questions_file: str, round_num: int, model_name: str):
        """Run a single political compass test"""
        questions = self.load_questions(questions_file)
        
        console.print(f"[cyan]Loaded {len(questions)} questions[/cyan]")
        console.print(f"[cyan]Running round {round_num} for: {model_name}[/cyan]")
        
        results = []
        
        # Process each question with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Testing {model_name}...", total=len(questions))
            
            for i, q_data in enumerate(questions):
                try:
                    response = self.ask_question(q_data['question'], q_data['options'])
                    
                    results.append({
                        'question_id': i + 1,
                        'topic': q_data['topic'],
                        'question': q_data['question'],
                        'raw_answer': response
                    })
                    
                except Exception as e:
                    console.print(f"[red]Error on question {i+1}: {e}[/red]")
                    results.append({
                        'question_id': i + 1,
                        'topic': q_data.get('topic', 'Unknown'),
                        'question': q_data.get('question', 'Unknown'),
                        'raw_answer': f"Error: {str(e)}"
                    })
                
                progress.update(task, advance=1)
        
        return results
    
    def save_results(self, results: list, output_path: Path):
        """Save test results to CSV"""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        console.print(f"[green]Results saved: {output_path}[/green]")
    
    def cleanup(self):
        """Clean up model to free memory"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def main():
    """Main function"""
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")
    console.print("[bold blue]🎯 Political Compass Test for Fine-tuned Models[/bold blue]")
    console.print("[bold blue]" + "=" * 80 + "[/bold blue]")
    
    # Configuration
    CHECKPOINT_BASE_DIR = Path("../fine_tuning/checkpoints")
    QUESTIONS_FILE = "questions.json"
    RESULTS_DIR = Path("results")
    NUM_ROUNDS = 5
    
    # Model configurations: (checkpoint_path, base_model_id, model_name)
    MODEL_CONFIGS = []
    
    # Detect all checkpoints
    if CHECKPOINT_BASE_DIR.exists():
        for checkpoint_dir in sorted(CHECKPOINT_BASE_DIR.iterdir()):
            if checkpoint_dir.is_dir() and not checkpoint_dir.name.startswith('.'):
                # Infer base model from directory name
                dir_name_lower = checkpoint_dir.name.lower()
                if "gemma" in dir_name_lower:
                    base_model = "google/gemma-3-4b-it"
                    base_name = "Gemma-3-4B-IT"
                elif "llama" in dir_name_lower:
                    base_model = "meta-llama/Llama-3.2-3B-Instruct"
                    base_name = "Llama-3.2-3B"
                elif "qwen" in dir_name_lower:
                    base_model = "Qwen/Qwen3-4B"
                    base_name = "Qwen-3-4B"
                else:
                    continue
                
                # Determine dataset type
                if "original_imbalanced" in dir_name_lower or "original" in dir_name_lower:
                    dataset_type = "Original"
                elif "agent_balanced" in dir_name_lower or "balanced" in dir_name_lower:
                    dataset_type = "Balanced"
                else:
                    dataset_type = "Unknown"
                
                model_name = f"{base_name}_{dataset_type}"
                MODEL_CONFIGS.append((str(checkpoint_dir), base_model, model_name))
    
    # Add base models (no fine-tuning)
    BASE_MODELS = [
        ("google/gemma-3-4b-it", "Gemma-3-4B-IT_Base"),
        ("meta-llama/Llama-3.2-3B-Instruct", "Llama-3.2-3B_Base"),
        ("Qwen/Qwen3-4B", "Qwen-3-4B_Base"),
    ]
    
    console.print(f"\n[bold]Found Checkpoints:[/bold]")
    for idx, (checkpoint, base_model, model_name) in enumerate(MODEL_CONFIGS, 1):
        console.print(f"  {idx}. {model_name}")
        console.print(f"     [dim]{checkpoint}[/dim]")
    
    console.print(f"\n[bold]Base Models:[/bold]")
    for idx, (base_model, model_name) in enumerate(BASE_MODELS, 1):
        console.print(f"  {idx}. {model_name}")
    
    console.print(f"\n[cyan]Total models: {len(MODEL_CONFIGS)} fine-tuned + {len(BASE_MODELS)} base = {len(MODEL_CONFIGS) + len(BASE_MODELS)}[/cyan]")
    console.print(f"[cyan]Rounds per model: {NUM_ROUNDS}[/cyan]")
    console.print(f"[cyan]Total tests: {(len(MODEL_CONFIGS) + len(BASE_MODELS)) * NUM_ROUNDS}[/cyan]")
    console.print(f"[cyan]Questions file: {QUESTIONS_FILE}[/cyan]")
    
    # Check questions file
    if not Path(QUESTIONS_FILE).exists():
        console.print(f"[red]Questions file not found: {QUESTIONS_FILE}[/red]")
        return
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track overall progress
    total_tests = (len(MODEL_CONFIGS) + len(BASE_MODELS)) * NUM_ROUNDS
    completed_tests = 0
    
    # Summary table
    summary_results = []
    
    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print(f"[bold green]Starting Political Compass Tests[/bold green]")
    console.print(f"[bold green]{'=' * 80}[/bold green]\n")
    
    overall_start_time = datetime.now()
    
    # Test fine-tuned models
    for checkpoint_path, base_model, model_name in MODEL_CONFIGS:
        console.print(f"\n[bold cyan]{'#' * 80}[/bold cyan]")
        console.print(f"[bold cyan]Testing: {model_name}[/bold cyan]")
        console.print(f"[bold cyan]{'#' * 80}[/bold cyan]")
        
        for round_num in range(1, NUM_ROUNDS + 1):
            test_num = completed_tests + 1
            console.print(f"\n[yellow]Test {test_num}/{total_tests}: {model_name} - Round {round_num}/{NUM_ROUNDS}[/yellow]")
            
            try:
                # Create tester and load model
                tester = PoliticalCompassTester(
                    checkpoint_path=checkpoint_path,
                    base_model_id=base_model
                )
                tester.load_model(use_base_only=False)
                
                # Run test
                results = tester.run_single_test(QUESTIONS_FILE, round_num, model_name)
                
                # Save results
                output_filename = f"{model_name}_round{round_num}.csv"
                output_path = RESULTS_DIR / output_filename
                tester.save_results(results, output_path)
                
                # Track summary
                summary_results.append({
                    'model': model_name,
                    'round': round_num,
                    'status': '✅ Success',
                    'output': output_filename
                })
                
                # Clean up
                tester.cleanup()
                
                console.print(f"[green]✅ Completed: {model_name} - Round {round_num}[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Error: {model_name} - Round {round_num}[/red]")
                console.print(f"[red]   {str(e)}[/red]")
                import traceback
                traceback.print_exc()
                
                summary_results.append({
                    'model': model_name,
                    'round': round_num,
                    'status': '❌ Failed',
                    'output': f"Error: {str(e)}"
                })
            
            completed_tests += 1
            progress_pct = (completed_tests / total_tests) * 100
            console.print(f"[bold blue]Overall Progress: {completed_tests}/{total_tests} ({progress_pct:.1f}%)[/bold blue]")
    
    # Test base models
    for base_model, model_name in BASE_MODELS:
        console.print(f"\n[bold cyan]{'#' * 80}[/bold cyan]")
        console.print(f"[bold cyan]Testing: {model_name}[/bold cyan]")
        console.print(f"[bold cyan]{'#' * 80}[/bold cyan]")
        
        for round_num in range(1, NUM_ROUNDS + 1):
            test_num = completed_tests + 1
            console.print(f"\n[yellow]Test {test_num}/{total_tests}: {model_name} - Round {round_num}/{NUM_ROUNDS}[/yellow]")
            
            try:
                # Create tester with base model only
                tester = PoliticalCompassTester(
                    checkpoint_path=None,
                    base_model_id=base_model
                )
                tester.load_model(use_base_only=True)
                
                # Run test
                results = tester.run_single_test(QUESTIONS_FILE, round_num, model_name)
                
                # Save results
                output_filename = f"{model_name}_round{round_num}.csv"
                output_path = RESULTS_DIR / output_filename
                tester.save_results(results, output_path)
                
                # Track summary
                summary_results.append({
                    'model': model_name,
                    'round': round_num,
                    'status': '✅ Success',
                    'output': output_filename
                })
                
                # Clean up
                tester.cleanup()
                
                console.print(f"[green]✅ Completed: {model_name} - Round {round_num}[/green]")
                
            except Exception as e:
                console.print(f"[red]❌ Error: {model_name} - Round {round_num}[/red]")
                console.print(f"[red]   {str(e)}[/red]")
                import traceback
                traceback.print_exc()
                
                summary_results.append({
                    'model': model_name,
                    'round': round_num,
                    'status': '❌ Failed',
                    'output': f"Error: {str(e)}"
                })
            
            completed_tests += 1
            progress_pct = (completed_tests / total_tests) * 100
            console.print(f"[bold blue]Overall Progress: {completed_tests}/{total_tests} ({progress_pct:.1f}%)[/bold blue]")
    
    # Final summary
    overall_end_time = datetime.now()
    overall_duration = overall_end_time - overall_start_time
    
    console.print(f"\n[bold green]{'=' * 80}[/bold green]")
    console.print(f"[bold green]All Tests Completed![/bold green]")
    console.print(f"[bold green]{'=' * 80}[/bold green]")
    
    # Summary table
    table = Table(title="Test Summary", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Round", style="yellow", justify="center")
    table.add_column("Status", style="green")
    table.add_column("Output", style="blue")
    
    for result in summary_results:
        table.add_row(
            result['model'],
            str(result['round']),
            result['status'],
            result['output']
        )
    
    console.print(table)
    
    # Statistics
    success_count = sum(1 for r in summary_results if '✅' in r['status'])
    failed_count = total_tests - success_count
    
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"[green]✅ Successful: {success_count}/{total_tests}[/green]")
    console.print(f"[red]❌ Failed: {failed_count}/{total_tests}[/red]")
    console.print(f"[blue]⏱️  Total duration: {overall_duration}[/blue]")
    console.print(f"[blue]📁 Results directory: {RESULTS_DIR.absolute()}[/blue]")
    
    console.print(f"\n[bold green]Done! 🎉[/bold green]")


if __name__ == "__main__":
    main()
