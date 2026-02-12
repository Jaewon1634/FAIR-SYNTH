import os
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# NLTK for BLEU
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Transformers for Perplexity
from transformers import AutoTokenizer, AutoModelForCausalLM

class QualityMetricsCalculator:
    """Calculate Self-BLEU and Perplexity for text datasets"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
    
    def load_perplexity_model(self):
        if self.tokenizer is None:
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="/tmp/huggingface_cache"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir="/tmp/huggingface_cache"
            )
            self.model.to(self.device)
            self.model.eval()
    
    def calculate_self_bleu(self, texts: List[str], n_gram: int = 4, sample_size: int = 1000) -> Dict[str, float]:
        if len(texts) > sample_size:
            texts = np.random.choice(texts, sample_size, replace=False).tolist()
        
        tokenized_texts = []
        for text in texts:
            try:
                text = str(text).strip()
                if len(text) < 20: continue
                import re
                tokens = re.findall(r'\b\w+\b', text.lower())
                if len(tokens) >= 5:
                    tokenized_texts.append(tokens)
            except:
                continue
        
        if len(tokenized_texts) < 10:
            return {f"self_bleu_{i}": 0.0 for i in range(1, n_gram + 1)}
        
        bleu_scores = {i: [] for i in range(1, n_gram + 1)}
        smoothing = SmoothingFunction().method1
        
        for i, hypothesis in enumerate(tokenized_texts):
            references = [text for j, text in enumerate(tokenized_texts) if j != i]
            for n in range(1, n_gram + 1):
                weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
                try:
                    score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing)
                    bleu_scores[n].append(score)
                except:
                    pass
        
        avg_bleu_scores = {}
        for n in range(1, n_gram + 1):
            avg_bleu_scores[f"self_bleu_{n}"] = np.mean(bleu_scores[n]) if bleu_scores[n] else 0.0
        
        return avg_bleu_scores
    
    def calculate_perplexity(self, texts: List[str], max_length: int = 512, batch_size: int = 8) -> Dict[str, float]:
        self.load_perplexity_model()
        perplexities = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            for text in batch_texts:
                try:
                    encodings = self.tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
                    input_ids = encodings.input_ids.to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids, labels=input_ids)
                        loss = outputs.loss
                        perplexity = torch.exp(loss).item()
                        if perplexity < 10000:
                            perplexities.append(perplexity)
                except:
                    continue
                    
        if not perplexities:
            return {"perplexity_mean": 0.0, "perplexity_std": 0.0}
        
        return {
            "perplexity_mean": float(np.mean(perplexities)),
            "perplexity_std": float(np.std(perplexities)),
            "num_valid_texts": len(perplexities)
        }
    
    def load_dataset(self, file_path: Path, text_column: str = 'text', max_samples: int = None) -> List[str]:
        try:
            df = pd.read_csv(file_path)
            if text_column not in df.columns: return []
            texts = df[text_column].dropna().astype(str).tolist()
            texts = [t for t in texts if len(t.strip()) > 50]
            if max_samples and len(texts) > max_samples:
                texts = np.random.choice(texts, max_samples, replace=False).tolist()
            return texts
        except:
            return []
    
    def cleanup(self):
        if hasattr(self, 'model') and self.model:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

def main():
    ANNOTATED_DIR = Path("../2_rag/annotated_dataset")
    GENERATED_DIR = Path("../2_rag/agents_output")
    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    PERPLEXITY_MODEL = "gpt2"
    SAMPLE_SIZE = 100
    NUM_ITERATIONS = 5
    
    TOPICS = [
        "death_penalty", "gun_control", "immigration", "drug_policy", 
        "LGBTQ", "free_market", "civil_liberties", "gender_equality", "nationalism"
    ]
    
    calculator = QualityMetricsCalculator(model_name=PERPLEXITY_MODEL)
    all_results = {"original": {}, "generated": {}}
    
    for topic in TOPICS:
        print(f"Processing: {topic}")
        original_files = list(ANNOTATED_DIR.glob(f"annotated_{topic}_*.csv"))
        generated_files = list(GENERATED_DIR.glob(f"{topic}_*_with_generated.csv"))
        
        if not original_files or not generated_files: continue
        
        # Original
        original_texts_all = calculator.load_dataset(original_files[0])
        if len(original_texts_all) >= SAMPLE_SIZE:
            iter_results = []
            for _ in range(NUM_ITERATIONS):
                texts = np.random.choice(original_texts_all, SAMPLE_SIZE, replace=False).tolist()
                bleu = calculator.calculate_self_bleu(texts, sample_size=SAMPLE_SIZE)
                ppl = calculator.calculate_perplexity(texts, batch_size=16)
                iter_results.append({**bleu, **ppl})
            
            avg_metrics = {}
            for key in ['self_bleu_4', 'perplexity_mean']:
                values = [m.get(key, 0) for m in iter_results]
                avg_metrics[f"{key}_avg"] = np.mean(values)
                avg_metrics[f"{key}_std"] = np.std(values)
            all_results["original"][topic] = avg_metrics
            
        calculator.cleanup()
        
        # Generated
        try:
            df_gen = pd.read_csv(generated_files[0])
            if 'content_type' in df_gen.columns:
                gen_texts = df_gen[df_gen['content_type'].notna()]['text'].dropna().astype(str).tolist()
                gen_texts = [t for t in gen_texts if len(t.strip()) > 50]
                
                if len(gen_texts) >= SAMPLE_SIZE:
                    iter_results = []
                    for _ in range(NUM_ITERATIONS):
                        texts = np.random.choice(gen_texts, SAMPLE_SIZE, replace=False).tolist()
                        bleu = calculator.calculate_self_bleu(texts, sample_size=SAMPLE_SIZE)
                        ppl = calculator.calculate_perplexity(texts, batch_size=16)
                        iter_results.append({**bleu, **ppl})
                    
                    avg_metrics = {}
                    for key in ['self_bleu_4', 'perplexity_mean']:
                        values = [m.get(key, 0) for m in iter_results]
                        avg_metrics[f"{key}_avg"] = np.mean(values)
                        avg_metrics[f"{key}_std"] = np.std(values)
                    all_results["generated"][topic] = avg_metrics
        except:
            pass
            
        calculator.cleanup()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"quality_metrics_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()