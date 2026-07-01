# Metric calculations for model evaluations

import numpy as np
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from src.validator import GroundednessValidator
from typing import List, Dict, Any

class EvaluationMetrics:
    """Class to manage, compute, and aggregate metrics on generated explanations."""
    
    def __init__(self):
        # Dynamically load evaluate modules
        try:
            self.rouge = evaluate.load("rouge")
        except Exception:
            self.rouge = None
            
        try:
            self.bleu = evaluate.load("sacrebleu")
        except Exception:
            self.bleu = None

        try:
            self.bertscore = evaluate.load("bertscore")
        except Exception:
            self.bertscore = None

    def compute_nlg_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Computes BLEU, ROUGE, and BERTScore."""
        results = {}
        
        # 1. ROUGE scores
        if self.rouge:
            try:
                rouge_results = self.rouge.compute(predictions=predictions, references=references)
                for k, v in rouge_results.items():
                    results[k] = float(v)
            except Exception as e:
                results["rouge_error"] = str(e)
        else:
            # Fallback manual calculation using rouge_score library
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            r1, r2, rl = [], [], []
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                r1.append(scores['rouge1'].fmeasure)
                r2.append(scores['rouge2'].fmeasure)
                rl.append(scores['rougeL'].fmeasure)
            results["rouge1"] = float(np.mean(r1))
            results["rouge2"] = float(np.mean(r2))
            results["rougeL"] = float(np.mean(rl))

        # 2. BLEU scores
        if self.bleu:
            try:
                bleu_results = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
                results["bleu"] = float(bleu_results["score"]) / 100.0  # Normalize to 0-1 range
            except Exception:
                pass
        
        if "bleu" not in results:
            # Fallback BLEU using nltk
            cc = SmoothingFunction()
            bleu_scores = []
            for pred, ref in zip(predictions, references):
                ref_tokens = ref.split()
                pred_tokens = pred.split()
                bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=cc.method1))
            results["bleu"] = float(np.mean(bleu_scores))

        # 3. BERTScore
        if self.bertscore:
            try:
                # Default to English (en) language
                bert_results = self.bertscore.compute(predictions=predictions, references=references, lang="en")
                results["bertscore_precision"] = float(np.mean(bert_results["precision"]))
                results["bertscore_recall"] = float(np.mean(bert_results["recall"]))
                results["bertscore_f1"] = float(np.mean(bert_results["f1"]))
            except Exception as e:
                results["bertscore_error"] = str(e)
                
        return results

    def compute_groundedness_metrics(self, cips: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Computes groundedness, length, and validation violation statistics."""
        violations_count = 0
        total_violations_recorded = 0
        total_length = 0
        violations_details = []

        for cip, pred in zip(cips, predictions):
            is_valid, violations = GroundednessValidator.validate(cip, pred)
            total_length += len(pred.split())
            if not is_valid:
                violations_count += 1
                total_violations_recorded += len(violations)
                violations_details.append({
                    "candidate_id": cip.get("candidate_id", "Unknown"),
                    "violations": violations,
                    "prediction": pred
                })

        num_samples = len(predictions)
        groundedness_rate = 1.0 - (violations_count / num_samples) if num_samples > 0 else 1.0

        return {
            "groundedness_rate": groundedness_rate,
            "average_word_count": total_length / num_samples if num_samples > 0 else 0,
            "total_violations_count": total_violations_recorded,
            "violations_percentage": (violations_count / num_samples) * 100.0 if num_samples > 0 else 0,
            "sample_violations": violations_details[:5]  # return first 5 for analysis reporting
        }
