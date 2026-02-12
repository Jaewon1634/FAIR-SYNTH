import json
import pandas as pd
from typing import List, Dict, Any, Tuple
from collections import Counter

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base import BaseAnalyzer
from core.types import Perspective


class LabelAnalyzer(BaseAnalyzer): 
    """Analyzes political labels using major voting across multiple GPT-4.1 persona annotations"""
    
    def __init__(self):
        self.gpt_cols = [
            'gpt-4.1_opp_left',
            'gpt-4.1_opp_right',
            'gpt-4.1_sup_left', 
            'gpt-4.1_sup_right'
        ]
        
    def analyze(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze political labels for each row using major voting
        
        For each sample:
        1. Collect Political and Stance labels/scores from 4 personas
        2. Apply majority voting for Political (Left/Center/Right)
        3. Apply majority voting for Stance (Support/Against/Neutral)
        4. Use average score to resolve ties (-0.2 to 0.2 → Center/Neutral)
        5. Generate final label as "Political-Stance" (e.g., "Left-Support")
        """
        results: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            row_result: Dict[str, Any] = {'id': row['id']}
            
            # Collect annotations from all 4 personas
            annotations = []
            for col in self.gpt_cols:
                pol_label, stance_label, pol_score, stance_score = self._extract_labels_and_scores(row[col])
                annotations.append((pol_label, stance_label, pol_score, stance_score))
            
            # Determine major political and stance labels
            major_political, major_stance = self._determine_major_labels(annotations)
            
            # Store results
            row_result['political_major'] = major_political
            row_result['stance_major'] = major_stance
            row_result['main_perspective'] = f"{major_political}-{major_stance}"
            
            # Store individual annotations for reference
            for i, (pol_label, stance_label, pol_score, stance_score) in enumerate(annotations):
                persona = self.gpt_cols[i].replace('gpt-4.1_', '')
                row_result[f'{persona}_political'] = pol_label
                row_result[f'{persona}_stance'] = stance_label
                row_result[f'{persona}_political_score'] = pol_score
                row_result[f'{persona}_stance_score'] = stance_score

            results.append(row_result)

        return results
    
    def _extract_labels_and_scores(self, json_str: str) -> Tuple[str, str, float, float]:
        """
        Extract political/stance labels and scores from GPT annotation
        
        Returns:
            (political_label, stance_label, political_score, stance_score)
        """
        try:
            data = json.loads(json_str)
            political_label = data.get('Political', {}).get('label', 'Undecided')
            stance_label = data.get('Stance', {}).get('label', 'Undecided')
            political_score = data.get('Political', {}).get('score', 0.0)
            stance_score = data.get('Stance', {}).get('score', 0.0)
            return political_label, stance_label, political_score, stance_score
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 'Undecided', 'Undecided', 0.0, 0.0
    
    def _determine_major_labels(self, annotations: List[Tuple[str, str, float, float]]) -> Tuple[str, str]:
        """
        Determine major political and stance labels using majority voting
        
        Args:
            annotations: List of (political_label, stance_label, political_score, stance_score)
        
        Returns:
            (major_political, major_stance)
        """
        # Separate votes and scores
        political_votes = [ann[0] for ann in annotations]
        stance_votes = [ann[1] for ann in annotations]
        political_scores = [ann[2] for ann in annotations]
        stance_scores = [ann[3] for ann in annotations]
        
        # Calculate average scores for tie-breaking
        avg_political_score = sum(political_scores) / len(political_scores) if political_scores else 0.0
        avg_stance_score = sum(stance_scores) / len(stance_scores) if stance_scores else 0.0
        
        # Major voting for political
        political_counter = Counter(political_votes)
        political_most_common = political_counter.most_common(2)
        
        if len(political_most_common) == 1:
            # Only one label exists
            major_political = political_most_common[0][0]
        elif political_most_common[0][1] > political_most_common[1][1]:
            # Clear majority (top vote count > second vote count)
            major_political = political_most_common[0][0]
        else:
            # Tie: use average score to decide
            if -0.2 <= avg_political_score <= 0.2:
                major_political = 'Center'
            elif avg_political_score < 0:
                major_political = 'Left'
            else:
                major_political = 'Right'
        
        # Major voting for stance
        stance_counter = Counter(stance_votes)
        stance_most_common = stance_counter.most_common(2)
        
        if len(stance_most_common) == 1:
            # Only one label exists
            major_stance = stance_most_common[0][0]
        elif stance_most_common[0][1] > stance_most_common[1][1]:
            # Clear majority (top vote count > second vote count)
            major_stance = stance_most_common[0][0]
        else:
            # Tie: use average score to decide
            if -0.2 <= avg_stance_score <= 0.2:
                major_stance = 'Neutral'
            elif avg_stance_score < 0:
                major_stance = 'Against'
            else:
                major_stance = 'Support'
        
        return major_political, major_stance
