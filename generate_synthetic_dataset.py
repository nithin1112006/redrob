# Synthetic dataset generator for Phase 2 explanation engine

import os
import json
import random
import argparse
from typing import Dict, Any, List
from src.utils import setup_logging, load_yaml, set_seed
from src.constants import DATASET_CONFIG_PATH

logger = setup_logging()

# Base pools for generating random but realistic candidate data
SKILL_POOL = [
    "Python", "Retrieval", "Vector Database", "Embeddings", "Ranking", "Evaluation Metrics",
    "SQL", "FastAPI", "Docker", "Kubernetes", "PyTorch", "Transformers", "LLMs", "NLP",
    "CI/CD", "Machine Learning", "System Design", "AWS", "Git", "Java", "Go", "Rust"
]

COMPANIES_POOL = ["Google", "Microsoft", "Meta", "Amazon", "Apple", "Netflix", "Stripe", "Spotify"]

NEGATIVE_SIGNALS_POOL = [
    "60 Day Notice", "30 Day Notice", "No Production Experience", "Lacks Scale Experience",
    "Higher Salary Expectation", "Location Constraint", "90 Day Notice"
]

POSITIVE_SIGNALS_POOL = [
    "Production Retrieval", "Leadership", "Open Source Contributor", "Ex-FAANG",
    "Fast Learner", "System Design Expert", "High Integrity Rating"
]

def generate_random_cip(candidate_id: str) -> Dict[str, Any]:
    """Generates a realistic Candidate Intelligence Package (CIP) with randomized attributes."""
    # Choose 3-6 random skills for must, preferred, and missing
    all_skills = random.sample(SKILL_POOL, k=random.randint(5, 9))
    must_count = random.randint(2, 3)
    pref_count = random.randint(1, 3)
    
    must_skills = all_skills[:must_count]
    preferred_skills = all_skills[must_count:must_count+pref_count]
    missing_skills = all_skills[must_count+pref_count:]
    
    pos_sigs = random.sample(POSITIVE_SIGNALS_POOL, k=random.randint(1, 3))
    # Inject some skills into positive signals
    pos_sigs.extend(random.sample(must_skills + preferred_skills, k=random.randint(1, 2)))
    pos_sigs = list(set(pos_sigs))
    
    neg_sigs = random.sample(NEGATIVE_SIGNALS_POOL, k=random.randint(0, 1))
    
    score = round(random.uniform(0.70, 0.99), 3)
    tech_fit = round(random.uniform(0.75, 0.99), 2)
    career_evidence = round(random.uniform(0.70, 0.98), 2)
    behavior = round(random.uniform(0.70, 0.98), 2)
    integrity = round(random.choice([0.8, 0.9, 1.0]), 1)
    confidence = round(random.uniform(0.80, 0.99), 2)
    rank = random.randint(1, 50)
    
    # Calculate feature contributions summing up to 100
    contribs = [random.randint(10, 40) for _ in range(4)]
    total = sum(contribs)
    contribs = [int((c / total) * 100) for c in contribs]
    # Adjust for rounding
    contribs[0] += (100 - sum(contribs))
    
    cip = {
        "candidate_id": candidate_id,
        "rank": rank,
        "score": score,
        "technical_fit": tech_fit,
        "career_evidence": career_evidence,
        "behavior": behavior,
        "integrity": integrity,
        "confidence": confidence,
        "positive_signals": pos_sigs,
        "negative_signals": neg_sigs,
        "jd_coverage": {
            "must": must_skills,
            "preferred": preferred_skills,
            "missing": missing_skills
        },
        "feature_contributions": {
            "career": contribs[0],
            "technical": contribs[1],
            "behavior": contribs[2],
            "integrity": contribs[3]
        }
    }
    return cip

def generate_explanation_text(cip: Dict[str, Any], style: str) -> str:
    """Generates ground truth recruiter explanation matching the input package."""
    must_skills = ", ".join(cip["jd_coverage"]["must"])
    pref_skills = ", ".join(cip["jd_coverage"]["preferred"])
    missing_skills = ", ".join(cip["jd_coverage"]["missing"])
    pos_sigs = ", ".join(cip["positive_signals"])
    
    neg_sig_text = ""
    if cip["negative_signals"]:
        neg_sig_text = f" However, {', '.join(cip['negative_signals'])} is noted."
        
    rank = cip["rank"]
    score = cip["score"]
    tech_fit = int(cip["technical_fit"] * 100)
    career_ev = int(cip["career_evidence"] * 100)
    confidence = int(cip["confidence"] * 100)
    
    # Template engine for the four explanation styles
    if style == "Recruiter":
        text = (
            f"Candidate {cip['candidate_id']} ranks highly at #{rank} with an overall score of {score}. "
            f"Demonstrates strong technical alignment ({tech_fit}% fit) with positive signals in {pos_sigs}. "
            f"Excellent coverage of core requirements like {must_skills} and preferred competencies like {pref_skills}."
            f"{neg_sig_text} Evaluation confidence stands at {confidence}%."
        )
    elif style == "Professional":
        text = (
            f"The evaluation profile for Candidate {cip['candidate_id']} indicates an overall match score of {score}, "
            f"placing them at rank position {rank}. Strong technical fit ({tech_fit}%) and career evidence ({career_ev}%) "
            f"are substantiated by expertise in {must_skills}. They offer additional capability in {pref_skills}, "
            f"although there is a lack of experience in {missing_skills}.{neg_sig_text} Confidence in this assessment is {confidence}%."
        )
    elif style == "Concise":
        text = (
            f"Rank #{rank} (Score: {score}). Strong {must_skills} and {pos_sigs} capabilities. "
            f"Lacks {missing_skills}.{neg_sig_text} Fit: {tech_fit}%. Confidence: {confidence}%."
        )
    elif style == "Technical":
        text = (
            f"Technical validation for candidate {cip['candidate_id']} indicates {tech_fit}% fit. "
            f"Core requirements met: {must_skills}. Secondary attributes verified: {pref_skills}. "
            f"Gaps identified: {missing_skills}. Strong performance markers found in {pos_sigs}. "
            f"Assessments indicate career contribution weight of {cip['feature_contributions']['career']}% "
            f"and technical weight of {cip['feature_contributions']['technical']}%.{neg_sig_text}"
        )
    else:
        text = f"Candidate profile matches requirements with score {score}."
        
    return text

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic instructions for Phase 2 SFT")
    parser.add_argument("--config", type=str, default=DATASET_CONFIG_PATH, help="Path to dataset config YAML")
    parser.add_argument("--num_samples", type=int, default=None, help="Force override count")
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    num_samples = args.num_samples or config["generation"]["num_samples"]
    output_dir = config["generation"]["output_dir"]
    styles = config["generation"]["styles"]
    seed = config["generation"].get("seed", 42)
    
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating {num_samples} synthetic candidates...")
    
    output_file = os.path.join(output_dir, "raw_synthetic_dataset.jsonl")
    
    with open(output_file, 'w') as f:
        for idx in range(num_samples):
            candidate_id = f"C{1000 + idx}"
            cip = generate_random_cip(candidate_id)
            style = random.choice(styles)
            explanation = generate_explanation_text(cip, style)
            
            record = {
                "candidate_package": cip,
                "style": style,
                "explanation": explanation
            }
            f.write(json.dumps(record) + "\n")
            
            if (idx + 1) % 2000 == 0:
                logger.info(f"Generated {idx + 1}/{num_samples} samples.")
                
    logger.info(f"Dataset generated and saved successfully to: {output_file}")

if __name__ == "__main__":
    main()
