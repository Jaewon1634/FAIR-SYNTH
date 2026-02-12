import os
import json
import torch
import logging
import psutil
try:
    import GPUtil
except Exception:
    GPUtil = None
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import time
from rich.console import Console
from rich.table import Table

console = Console()

@dataclass
class ModelConfig:
    """Model configuration dataclass"""
    model_name: str
    model_id: str
    max_length: int = 4096
    batch_size: int = 2
    learning_rate: float = 3e-5
    num_epochs: int = 3
    warmup_steps: int = 300
    save_steps: int = 200
    eval_steps: int = 200
    gradient_accumulation_steps: int = 8
    fp16: bool = False
    bf16: bool = True
    use_lora: bool = False
    use_qlora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    logging_first_step: bool = True
    objective: str = "causal_lm"

class SystemMonitor:
    """System resource monitoring class"""
    
    @staticmethod
    def get_gpu_info():
        """Get GPU information"""
        try:
            if GPUtil is None:
                return []
            gpus = GPUtil.getGPUs()
            gpu_info = []
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': f"{gpu.memoryTotal}MB",
                    'memory_used': f"{gpu.memoryUsed}MB",
                    'memory_free': f"{gpu.memoryFree}MB",
                    'utilization': f"{gpu.load*100:.1f}%",
                    'temperature': f"{gpu.temperature}°C"
                })
            return gpu_info
        except:
            return []
    
    @staticmethod
    def get_cpu_memory_info():
        """Get CPU and memory information"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return {
            'cpu_usage': f"{cpu_percent:.1f}%",
            'memory_total': f"{memory.total / (1024**3):.1f}GB",
            'memory_used': f"{memory.used / (1024**3):.1f}GB",
            'memory_available': f"{memory.available / (1024**3):.1f}GB",
            'memory_percent': f"{memory.percent:.1f}%"
        }
    
    @staticmethod
    def print_system_info():
        """Print system information"""
        table = Table(title="System Resources")
        table.add_column("Category", style="cyan")
        table.add_column("Info", style="green")
        
        # CPU/Memory info
        sys_info = SystemMonitor.get_cpu_memory_info()
        table.add_row("CPU Usage", sys_info['cpu_usage'])
        table.add_row("Memory Usage", sys_info['memory_percent'])
        table.add_row("Available Memory", sys_info['memory_available'])
        
        # GPU info
        gpu_info = SystemMonitor.get_gpu_info()
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                table.add_row(f"GPU {i} ({gpu['name']})", 
                            f"Usage: {gpu['utilization']}, Memory: {gpu['memory_used']}/{gpu['memory_total']}")
        else:
            table.add_row("GPU", "Not available")
        
        console.print(table)

class DatasetHandler:
    """Dataset handling utility class"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str):
        """Save data to JSONL file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

class TrainingLogger:
    """Training logger class"""
    
    def __init__(self, log_dir: str, model_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log file
        log_file = self.log_dir / f"{model_name}_{int(time.time())}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(model_name)
    
    def log_training_start(self, config: Dict[str, Any]):
        """Log training start"""
        self.logger.info(f"Training started: {config}")
        
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start"""
        self.logger.info(f"Epoch {epoch+1}/{total_epochs} started")
        
    def log_step(self, step: int, loss: float, learning_rate: float):
        """Log training step"""
        self.logger.info(f"Step {step} - Loss: {loss:.4f}, LR: {learning_rate:.2e}")
        
    def log_evaluation(self, metrics: Dict[str, float]):
        """Log evaluation results"""
        self.logger.info(f"Evaluation results: {metrics}")
        
    def log_training_complete(self, total_time: float):
        """Log training completion"""
        self.logger.info(f"Training completed. Total time: {total_time:.2f}s")

def setup_device():
    """Setup compute device"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]CUDA available: {torch.cuda.get_device_name()}[/green]")
    else:
        device = torch.device("cpu")
        console.print("[yellow]Running on CPU mode[/yellow]")
    
    return device

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to JSON file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_time(seconds: float) -> str:
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
