import os
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from src.utils import setup_logging, load_yaml
from src.constants import TRAINING_CONFIG_PATH, INFERENCE_CONFIG_PATH
try:
    from src.inference_engine import InferenceEngine
except ImportError:
    InferenceEngine = None


logger = setup_logging()

# Column synonym lookup table to support any input format
COLUMN_MAPPINGS = {
    "candidate_id": ["candidate_id", "id", "candidate id", "cand_id", "candidate", "cand"],
    "rank": ["rank", "ranking", "position", "pos"],
    "score": ["score", "match_score", "total_score", "points"],
    "title": ["title", "job_title", "role", "job", "position_title"],
    "company": ["company", "org", "organization", "employer"],
    "technical_fit": ["technical_fit", "technical fit", "tech_fit", "technical", "fit"],
    "career_evidence": ["career_evidence", "career evidence", "career", "experience", "experience_years"],
    "behavior": ["behavior", "behaviour", "behavioral", "behavior_score"],
    "integrity": ["integrity", "integrity_score", "ethics"]
}

def map_headers(columns: List[str]) -> Dict[str, str]:
    """Maps input columns to normalized Candidate Intelligence Package fields."""
    mapped = {}
    for col in columns:
        col_lower = str(col).lower().strip()
        matched = False
        for standard_name, synonyms in COLUMN_MAPPINGS.items():
            if col_lower in synonyms:
                mapped[standard_name] = col
                matched = True
                break
        if not matched:
            # Fallback fuzzy substring matching
            for standard_name, synonyms in COLUMN_MAPPINGS.items():
                if any(syn in col_lower for syn in synonyms):
                    mapped[standard_name] = col
                    break
    return mapped

def build_cip_from_row(row: pd.Series, mapped_cols: Dict[str, str]) -> Dict[str, Any]:
    """Assembles a valid Candidate Intelligence Package from a spreadsheet row."""
    def get_val(key, default=0.0):
        if key in mapped_cols:
            val = row[mapped_cols[key]]
            if pd.isna(val):
                return default
            return val
        return default

    candidate_id = str(row[mapped_cols["candidate_id"]]) if "candidate_id" in mapped_cols else "CAND_UNKNOWN"
    rank = int(get_val("rank", 0))
    score = float(get_val("score", 0.0))
    
    # Normalize score if it is represented on 0-100 scale instead of 0-1
    if score > 1.0:
        score = round(score / 100.0, 3)

    tech_fit = float(get_val("technical_fit", 0.8))
    if tech_fit > 1.0:
        tech_fit = round(tech_fit / 100.0, 2)
        
    career = float(get_val("career_evidence", 0.8))
    if career > 1.0:
        career = round(career / 100.0, 2)
        
    behavior = float(get_val("behavior", 0.8))
    if behavior > 1.0:
        behavior = round(behavior / 100.0, 2)
        
    integrity = float(get_val("integrity", 1.0))
    if integrity > 1.0:
        integrity = round(integrity / 100.0, 2)
        
    title = str(row[mapped_cols["title"]]) if "title" in mapped_cols else "Engineer"
    company = str(row[mapped_cols["company"]]) if "company" in mapped_cols else "Company"

    # Extrapolate signals from metadata
    pos_sigs = ["Production Experience"]
    if title:
        pos_sigs.append(title)
    if company:
        pos_sigs.append(f"Ex-{company}")
        
    # Extrapolate JD coverage
    must_skills = []
    if title:
        must_skills.extend(title.split()[:2])

    cip = {
        "candidate_id": candidate_id,
        "rank": rank,
        "score": score,
        "technical_fit": tech_fit,
        "career_evidence": career,
        "behavior": behavior,
        "integrity": integrity,
        "confidence": round((tech_fit + career + behavior) / 3.0, 2),
        "positive_signals": pos_sigs,
        "negative_signals": [],
        "jd_coverage": {
            "must": must_skills,
            "preferred": ["System Design"],
            "missing": []
        },
        "feature_contributions": {
            "career": 40,
            "technical": 40,
            "behavior": 20,
            "integrity": 0
        }
    }
    return cip

def main():
    parser = argparse.ArgumentParser(description="Adaptive Batch Processor for Candidate Excel/CSV data")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input Excel (.xlsx) or CSV file")
    parser.add_argument("--output_file", type=str, default="./outputs/reasoning_report.xlsx", help="Path to save output Excel file")
    parser.add_argument("--style", type=str, default="Recruiter", help="Explanation style: Recruiter, Professional, Concise, Technical")
    parser.add_argument("--train_config", type=str, default=TRAINING_CONFIG_PATH, help="Path to training config YAML")
    parser.add_argument("--inference_config", type=str, default=INFERENCE_CONFIG_PATH, help="Path to inference config YAML")
    parser.add_argument("--adapter_path", type=str, default=None, help="LoRA adapter directory")
    args = parser.parse_args()
    
    # 1. Load Data
    ext = os.path.splitext(args.input_file)[-1].lower()
    logger.info(f"Loading input file: {args.input_file}...")
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(args.input_file)
    elif ext == ".csv":
        df = pd.read_csv(args.input_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        
    logger.info(f"Columns found: {list(df.columns)}")
    
    # 2. Map Column Headers
    mapped_cols = map_headers(df.columns)
    logger.info(f"Mapped headers: {mapped_cols}")
    
    if "candidate_id" not in mapped_cols:
        raise ValueError("Could not find candidate identifier column (e.g. candidate_id, ID, Name).")
        
    # 3. Load Model Inference Engine
    train_conf = load_yaml(args.train_config)
    inf_conf = load_yaml(args.inference_config)
    
    # Enforce offline setup if configured
    if inf_conf.get("offline", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
    adapter_path = args.adapter_path
    if not adapter_path:
        adapter_path = os.path.join(train_conf["training"]["output_dir"], "final_adapter")
        if not os.path.exists(adapter_path):
            adapter_path = None
            
    base_model = train_conf["model"]["base_model_name"]
    
    engine = None
    try:
        engine = InferenceEngine(
            base_model_path=base_model,
            adapter_path=adapter_path,
            inference_config=inf_conf
        )
    except Exception as e:
        logger.warning(f"Could not load inference engine (possibly offline without cached weights). Using fallback template-based generation: {e}")

    # 4. Process Rows
    reasoning_1_list = []
    reasoning_2_list = []
    reasoning_3_list = []
    sentence_reason_list = []
    logger.info("Generating recruiter explanations...")
    
    import re
    for idx, row in df.iterrows():
        cip = build_cip_from_row(row, mapped_cols)
        
        # Extract and clean role title
        raw_title = str(row[mapped_cols["title"]]) if "title" in mapped_cols else "Engineer"
        role = re.sub(r'^\d+(?:\.\d+)?\s+', '', raw_title).strip()
        
        # Calculate metrics matching user formula
        career_yrs = cip["career_evidence"] * 10
        tech_skills = int(cip["technical_fit"] * 10)
        resp_rate = cip["behavior"]
        
        # 1. Store the 3 split reasoning columns
        reasoning_1_list.append(f"{role} with {career_yrs:.1f} yrs")
        reasoning_2_list.append(f"{tech_skills} AI core skills")
        reasoning_3_list.append(f"response rate {resp_rate:.2f}.")
        
        # 2. Generate the detailed sentence reasoning (model or fallback)
        if engine is not None:
            explanation, _ = engine.generate_explanation(cip, style=args.style)
        else:
            explanation = (
                f"Candidate {cip['candidate_id']} is ranked highly at position {cip['rank']} "
                f"with a score of {cip['score']}. They demonstrate excellent technical alignment "
                f"({int(cip['technical_fit']*100)}% fit) and strong career evidence "
                f"({int(cip['career_evidence']*100)}%)."
            )
        sentence_reason_list.append(explanation)
        
    # 5. Build Output DataFrame and save
    out_df = pd.DataFrame({
        "candidate_id": df[mapped_cols["candidate_id"]],
        "rank": df[mapped_cols["rank"]] if "rank" in mapped_cols else range(1, len(df)+1),
        "score": df[mapped_cols["score"]] if "score" in mapped_cols else 0.0,
        "reasoning_1": reasoning_1_list,
        "reasoning_2": reasoning_2_list,
        "reasoning_3": reasoning_3_list,
        "sentence_reason": sentence_reason_list
    })
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Write using ExcelWriter and openpyxl to merge header cells D1:F1
    writer = pd.ExcelWriter(args.output_file, engine="openpyxl")
    out_df.to_excel(writer, index=False, sheet_name="Sheet1")
    
    # Access openpyxl worksheet objects
    workbook = writer.book
    worksheet = writer.sheets["Sheet1"]
    
    # Merge D1, E1, F1 (columns 4, 5, 6) into a single header "Reason"
    worksheet.merge_cells("D1:F1")
    worksheet["D1"] = "Reason"
    
    # Apply center alignment to the merged Reason header
    from openpyxl.styles import Alignment
    worksheet["D1"].alignment = Alignment(horizontal="center")
    
    writer.close()
    logger.info(f"Report successfully saved to Excel file: {args.output_file}")

if __name__ == "__main__":
    main()
