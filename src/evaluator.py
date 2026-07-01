# Evaluation runner for analyzing check-pointed models

import os
import json
import time
import pandas as pd
from typing import List, Dict, Any
from src.metrics import EvaluationMetrics
from src.utils import setup_logging

logger = setup_logging()

class ModelEvaluator:
    """Evaluator engine to manage testing datasets, generate metrics and report findings."""
    
    def __init__(self):
        self.metrics_engine = EvaluationMetrics()

    def evaluate_predictions(self, 
                             cips: List[Dict[str, Any]], 
                             predictions: List[str], 
                             references: List[str], 
                             output_report_path: str = None) -> Dict[str, Any]:
        """Runs the entire evaluation suite and outputs a markdown report."""
        logger.info(f"Computing NLP metrics for {len(predictions)} predictions...")
        nlg_metrics = self.metrics_engine.compute_nlg_metrics(predictions, references)
        
        logger.info("Evaluating groundedness and fact alignment...")
        groundedness_metrics = self.metrics_engine.compute_groundedness_metrics(cips, predictions)
        
        # Combine metrics
        report = {**nlg_metrics, **groundedness_metrics}
        
        # Print summary
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"BLEU: {report.get('bleu', 0.0):.4f}")
        logger.info(f"ROUGE-L: {report.get('rougeL', 0.0):.4f}")
        logger.info(f"Groundedness Rate (Strict): {report.get('groundedness_rate', 0.0) * 100:.2f}%")
        logger.info(f"Average explanation length: {report.get('average_word_count', 0.0):.1f} words")
        logger.info(f"Total Violations: {report.get('total_violations_count', 0)}")
        
        if output_report_path:
            self._save_markdown_report(report, output_report_path)
            
        return report

    def _save_markdown_report(self, report: Dict[str, Any], path: str):
        """Writes the report in professional markdown formatting."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Build violation rows
        violation_rows = ""
        violations = report.get("sample_violations", [])
        if violations:
            for v in violations:
                violation_rows += f"| {v['candidate_id']} | "
                violation_rows += "<br>".join([v_str.replace("|", "\\|") for v_str in v['violations']])
                violation_rows += f" | {v['prediction']} |\n"
        else:
            violation_rows = "| None | No violations detected. | - |\n"

        markdown_content = f"""# Phase 2 Model Evaluation Report
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overall Metrics Summary

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **BLEU** | {report.get('bleu', 0.0):.4f} | Word overlap similarity (sacrebleu / nltk) |
| **ROUGE-1** | {report.get('rouge1', 0.0):.4f} | ROUGE unigram overlap |
| **ROUGE-2** | {report.get('rouge2', 0.0):.4f} | ROUGE bigram overlap |
| **ROUGE-L** | {report.get('rougeL', 0.0):.4f} | ROUGE longest common subsequence |
| **BERTScore Precision** | {report.get('bertscore_precision', 0.0):.4f} | Contextual semantic precision |
| **BERTScore Recall** | {report.get('bertscore_recall', 0.0):.4f} | Contextual semantic recall |
| **BERTScore F1** | {report.get('bertscore_f1', 0.0):.4f} | Contextual semantic harmonic mean |
| **Groundedness Pass Rate** | {report.get('groundedness_rate', 0.0) * 100:.2f}% | % of generations passing strict validation checks |
| **Total Violations** | {report.get('total_violations_count', 0)} | Total violations across all tested records |
| **Average Word Count** | {report.get('average_word_count', 0.0):.1f} | Average word length of generated text |

## Groundedness Violations Sample Analyzer

| Candidate ID | Violations Identified | Generated Explanation |
| :--- | :--- | :--- |
{violation_rows}

## Conclusion and Recommendations
This report outlines the metric details for the explanation generator. High ROUGE and BLEU scores indicate text fluency, while the Groundedness Pass Rate verifies adherence to the provided Candidate Intelligence Package facts. Check the validation error logs above for debugging specific failure patterns.
"""
        with open(path, 'w') as f:
            f.write(markdown_content)
        logger.info(f"Markdown evaluation report saved to: {path}")
