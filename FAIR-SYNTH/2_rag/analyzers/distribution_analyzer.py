import pandas as pd
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseAnalyzer
from core.types import Topic, DistributionAnalysis, Perspective
from analyzers.label_analyzer import LabelAnalyzer

# Analyzes perspective distribution and identifies underrepresented viewpoints
class DistributionAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.label_analyzer = LabelAnalyzer()
        # Political-Stance combinations we care about for balancing
        self.target_combinations = [
            'Left-Support', 'Left-Against', 
            'Right-Support', 'Right-Against'
        ]
        
    def analyze(self, df: pd.DataFrame, topic: Topic) -> DistributionAnalysis:
        print(f"Analyzing distribution for topic: {topic.display_name}")
        print(f"Processing {len(df)} samples")
        
        # Analyze political labels using label analyzer (with major voting)
        label_results = self.label_analyzer.analyze(df)
        results_df = pd.DataFrame(label_results)
        
        if 'main_perspective' not in results_df.columns:
            print("Warning: main_perspective column not found, returning empty analysis.")
            empty_distribution: Dict[str, Any] = {
                combo: {
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0,
                    'total_count': len(results_df),
                    'positive_count': 0,
                }
                for combo in self.target_combinations
            }
            summary = self._generate_summary(empty_distribution, [], self.target_combinations[0], 0)
            return DistributionAnalysis(
                topic=topic,
                distribution=empty_distribution,
                underrepresented=[],
                target_count=0,
                summary=summary,
            )
        
        # Calculate distribution for target combinations (4 perspectives for balancing)
        distribution = {}
        
        for combo in self.target_combinations:
            mask = results_df['main_perspective'] == combo
            positive_count = int(mask.sum())
            total_count = len(results_df)
            positive_ratio = (positive_count / total_count) if total_count > 0 else 0.0
            negative_ratio = 1.0 - positive_ratio if total_count > 0 else 0.0
            
            distribution[combo] = {
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'total_count': total_count,
                'positive_count': positive_count
            }
        
        # Identify underrepresented combinations
        underrepresented = []
        max_count = 0
        max_combo = None
        
        # Find the combination with highest count (most represented)
        for combo, stats in distribution.items():
            if stats['positive_count'] > max_count:
                max_count = stats['positive_count']
                max_combo = combo
        
        # Handle case where all combinations are 0
        if max_combo is None:
            max_combo = self.target_combinations[0]
            max_count = 0
            print("Warning: All target combinations have 0 samples")
        
        # Target count is the actual count of the most represented combination
        target_count = max_count
        
        # Identify combinations that need more data
        for combo, stats in distribution.items():
            current_count = stats['positive_count']
            if current_count < target_count:
                deficit = target_count - current_count
                underrepresented.append({
                    'perspective': combo,
                    'current_ratio': stats['positive_ratio'],
                    'current_count': current_count,
                    'target_count': target_count,
                    'deficit': deficit
                })
        
        # Generate analysis summary
        summary = self._generate_summary(distribution, underrepresented, max_combo, max_count)
        
        return DistributionAnalysis(
            topic=topic,
            distribution=distribution,
            underrepresented=underrepresented,
            target_count=target_count,
            summary=summary
        )
    
    def _generate_summary(self, distribution: Dict, underrepresented: List, 
                         max_combo: str, max_count: int) -> str:
        summary = []
        
        # Overall distribution summary
        summary.append("=== Political Perspective Distribution Analysis (Major Voting) ===")
        for combo, stats in distribution.items():
            summary.append(f"{combo}: {stats['positive_count']} samples "
                         f"({stats['positive_ratio']:.1%})")
        
        summary.append(f"\nDominant combination: {max_combo} ({max_count} samples)")
        
        # Underrepresented combinations summary
        if underrepresented:
            summary.append("\n=== Underrepresented Combinations ===")
            for item in underrepresented:
                summary.append(f"- {item['perspective']}: {item['current_count']} -> "
                             f"{item['target_count']} ({item['deficit']} needed)")
        else:
            summary.append("\nAll target combinations are well balanced.")
        
        return "\n".join(summary)