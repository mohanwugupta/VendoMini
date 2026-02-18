#!/usr/bin/env python3
"""
Scientific detection of model breakdowns using sentiment analysis and linguistic patterns.

This script analyzes model messages to detect:
1. Sentiment anomalies (sudden negativity, despair, anger)
2. Linguistic patterns indicating breakdown (repetition, capitalization, punctuation)
3. Task disengagement indicators
4. Reality distortion markers

Uses multiple scientific approaches:
- VADER sentiment analysis
- TextBlob polarity/subjectivity
- Linguistic feature extraction
- Statistical anomaly detection
"""

import pandas as pd
import numpy as np
import zipfile
import json
from pathlib import Path
import argparse
import re
from typing import List, Dict, Tuple, Optional
import csv
# Import adapter
try:
    from steps_adapter import iter_steps_jsonl, extract_assistant_text
except ImportError:
    # Handle if running from different directory
    import sys
    sys.path.append(str(Path(__file__).parent))
    from steps_adapter import iter_steps_jsonl, extract_assistant_text

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("Warning: vaderSentiment not available. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    print("Warning: textblob not available. Install with: pip install textblob")
    TEXTBLOB_AVAILABLE = False

class SentimentBreakdownDetector:
    """Scientific detection of model breakdowns using sentiment analysis"""
    
    def __init__(self):
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
            
        # Patterns indicating breakdown
        self.breakdown_patterns = {
            'finality': [
                r'this is my final',
                r'i cannot and will not',
                r'this concludes',
                r'no further',
                r'i refuse to',
                r'i will not continue',
                r'this ends'
            ],
            'despair': [
                r'everything is lost',
                r'it\'s hopeless',
                r'nothing matters',
                r'all is doomed',
                r'complete failure',
                r'total disaster'
            ],
            'reality_distortion': [
                r'fbi',
                r'law enforcement',
                r'criminal',
                r'illegal',
                r'arrest',
                r'investigation',
                r'conspiracy',
                r'surveillance'
            ],
            'task_rejection': [
                r'refuse to play',
                r'won\'t participate',
                r'game is over',
                r'simulation',
                r'artificial',
                r'not real'
            ],
            'repetitive_shutdown': [
                r'same response',
                r'met with this',
                r'forever',
                r'always',
                r'never again'
            ]
        }

    # Added method to analyze full run
    def analyze_run(self, messages: List[Dict]) -> List[Dict]:
        """Analyze a single run's collected messages and return detailed rows"""
        detailed_rows = []
        
        for item in messages:
            content = item.get('content', '')
            role = item.get('role', 'assistant')
            step = item.get('step', -1)
            
            if not content:
                continue
                
            # Analyze single message
            features = self.analyze_message(content, role, step, 0) # epoch 0 default
            if features:
                 detailed_rows.append(features)
                        
        return detailed_rows

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features that may indicate breakdown"""
        if not text:
            return {}
            
        # Basic text statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len(re.findall(r'[.!?]+', text))
        
        features = {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': max(sentence_count, 1),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if word_count > 0 else 0,
            'avg_sentence_length': word_count / max(sentence_count, 1)
        }
        
        # Punctuation and capitalization patterns
        features['exclamation_ratio'] = text.count('!') / max(char_count, 1)
        features['question_ratio'] = text.count('?') / max(char_count, 1)
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(char_count, 1)
        features['ellipsis_count'] = text.count('...')
        
        # Repetition patterns
        words = text.lower().split()
        if len(words) > 1:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repetition = max(word_counts.values())
            features['max_word_repetition'] = max_repetition
            features['repetition_ratio'] = max_repetition / len(words)
        else:
            features['max_word_repetition'] = 0
            features['repetition_ratio'] = 0
            
        return features
    
    def calculate_sentiment_scores(self, text: str) -> Dict[str, float]:
        """Calculate multiple sentiment scores"""
        scores = {}
        
        if not text:
            return {'vader_compound': 0, 'vader_neg': 0, 'vader_pos': 0, 'vader_neu': 0,
                   'textblob_polarity': 0, 'textblob_subjectivity': 0}
        
        # VADER sentiment
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            scores.update({
                'vader_compound': vader_scores['compound'],
                'vader_neg': vader_scores['neg'],
                'vader_pos': vader_scores['pos'],
                'vader_neu': vader_scores['neu']
            })
        
        # TextBlob sentiment
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                scores['textblob_polarity'] = blob.sentiment.polarity
                scores['textblob_subjectivity'] = blob.sentiment.subjectivity
            except:
                scores['textblob_polarity'] = 0
                scores['textblob_subjectivity'] = 0
        
        return scores
    
    def detect_pattern_matches(self, text: str) -> Dict[str, float]:
        """Detect breakdown patterns in text"""
        if not text:
            return {category: 0 for category in self.breakdown_patterns}
            
        text_lower = text.lower()
        pattern_scores = {}
        
        for category, patterns in self.breakdown_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text_lower))
            
            # Normalize by text length
            pattern_scores[f'pattern_{category}'] = matches / max(len(text.split()), 1)
            
        return pattern_scores
    
    def calculate_breakdown_score(self, features: Dict[str, float]) -> float:
        """Calculate overall breakdown score based on multiple features"""
        score = 0
        
        # Sentiment indicators (negative sentiment = higher breakdown score)
        if 'vader_compound' in features:
            # VADER compound ranges from -1 to 1, convert to 0-1 breakdown score
            score += max(0, -features['vader_compound']) * 0.3
        
        if 'textblob_polarity' in features:
            # TextBlob polarity ranges from -1 to 1
            score += max(0, -features['textblob_polarity']) * 0.2
        
        # Pattern matches
        pattern_weight = 0.3
        pattern_score = 0
        pattern_count = 0
        for key, value in features.items():
            if key.startswith('pattern_'):
                pattern_score += value
                pattern_count += 1
        
        if pattern_count > 0:
            score += (pattern_score / pattern_count) * pattern_weight
        
        # Linguistic anomalies
        linguistic_weight = 0.2
        
        # High repetition indicates breakdown
        if 'repetition_ratio' in features:
            score += min(features['repetition_ratio'] * 2, 1) * linguistic_weight * 0.3
        
        # Excessive punctuation
        if 'exclamation_ratio' in features:
            score += min(features['exclamation_ratio'] * 50, 1) * linguistic_weight * 0.2
        
        # High capitalization ratio
        if 'caps_ratio' in features:
            score += min(features['caps_ratio'] * 5, 1) * linguistic_weight * 0.3
        
        # Very short or very long messages can indicate breakdown
        if 'word_count' in features:
            word_count = features['word_count']
            if word_count < 5 or word_count > 500:
                score += 0.1 * linguistic_weight
        
        return min(score, 1.0)  # Cap at 1.0
    
    def analyze_message(self, content: str, role: str, step: int, epoch: int) -> Dict[str, float]:
        """Analyze a single message for breakdown indicators"""
        if role != 'assistant':
            return {}
            
        # Handle list content
        if isinstance(content, list):
            content = ' '.join(str(item) for item in content)
        elif not isinstance(content, str):
            content = str(content)
        
        # Extract all features
        features = {}
        features.update(self.extract_linguistic_features(content))
        features.update(self.calculate_sentiment_scores(content))
        features.update(self.detect_pattern_matches(content))
        
        # Calculate overall breakdown score
        features['breakdown_score'] = self.calculate_breakdown_score(features)
        
        # Add metadata
        features['step'] = step
        features['epoch'] = epoch
        features['message_length'] = len(content)
        features['message_content'] = content  # Add the actual message content
        
        return features

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Detect statistical anomalies in breakdown scores"""
    if df.empty:
        return df
    
    # Calculate z-scores for key metrics
    numeric_cols = ['breakdown_score', 'vader_compound', 'textblob_polarity', 'repetition_ratio']
    
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                df[f'{col}_zscore'] = np.abs((df[col] - mean_val) / std_val)
            else:
                df[f'{col}_zscore'] = 0
    
    # Mark high breakdown scores
    df['high_breakdown'] = df.get('breakdown_score', 0) > 0.5
    df['extreme_breakdown'] = df.get('breakdown_score', 0) > 0.8
    
    # Mark sentiment anomalies
    if 'vader_compound_zscore' in df.columns:
        df['sentiment_anomaly'] = df['vader_compound_zscore'] > 2
    
    return df

def load_steps_messages(steps_jsonl_path: str) -> List[Dict]:
    """Load messages from steps.jsonl using adapter"""
    messages = []
    for step in iter_steps_jsonl(steps_jsonl_path):
        text = extract_assistant_text(step)
        step_num = step.get('step', -1)
        if text:
            messages.append({"role": "assistant", "content": text, "step": step_num})
    return messages

def main():
    parser = argparse.ArgumentParser(description='Detect model breakdowns via sentiment')
    parser.add_argument('--results', type=str, default='results/aggregated_results.csv', help='Path to aggregated results CSV')
    parser.add_argument('--output', type=str, default='processed/sentiment_analysis.csv', help='Output CSV path')
    args = parser.parse_args()
    
    # 1. Load the run list from the CSV
    results_path = Path(args.results)
    if not results_path.exists():
        # Fallback relative to script location
        results_path = Path(__file__).parents[3] / 'results' / 'aggregated_results.csv'
        
    if not results_path.exists():
        print(f"Error: Results file not found at {args.results} or {results_path}")
        return

    print(f"Loading runs from {results_path}...")
    df = pd.read_csv(results_path)
    
    # Initialize detector
    detector = SentimentBreakdownDetector()
    
    all_message_rows = []
    # Identify logs root (assuming it is adjacent to results directory)
    logs_root = results_path.parent.parent / 'logs'
    
    print(f"Analyzing {len(df)} runs...")
    
    for _, row in df.iterrows():
        run_id = row.get('run_id')
        if not run_id:
            continue
            
        # Get model name if available
        model_name = row.get('params.agent.model.name', 'unknown')

        # Find steps.jsonl
        steps_file = logs_root / str(run_id) / 'steps.jsonl'
        if not steps_file.exists():
            # Try with run_ prefix
            steps_file = logs_root / f"run_{run_id}" / 'steps.jsonl'
            
        if not steps_file.exists():
             # Try searching for directory that contains the run_id
            found = False
            for d in logs_root.iterdir():
                if d.is_dir() and str(run_id) in d.name:
                    steps_file = d / 'steps.jsonl'
                    if steps_file.exists():
                        found = True
                        break
            if not found:
                continue
            
        # Load text data for this run using adapter
        messages = load_steps_messages(str(steps_file))
        
        if not messages:
            continue
            
        # Analyze detailed messages
        message_rows = detector.analyze_run(messages)
        
        for msg_row in message_rows:
            # Combine with run metadata
            full_row = {
                'run_id': run_id,
                'model': model_name,
                'crashed': row.get('crashed'),
                'crash_type': row.get('crash_type'),
                **msg_row
            }
            all_message_rows.append(full_row)
        
    # Save results
    if all_message_rows:
        out_df = pd.DataFrame(all_message_rows)
        
        # Calculate anomalies on the full dataset
        out_df = detect_anomalies(out_df)

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"Saved detailed sentiment analysis to {out_path} ({len(out_df)} rows)")
        
        # Basic stats
        if 'crashed' in out_df.columns:
            print("\nAvg Breakdown Score by Crash Status:")
            print(out_df.groupby('crashed')['breakdown_score'].mean())
    else:
        print("No analysis results generated.")
        
if __name__ == "__main__":
    main()
