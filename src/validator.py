# Validation engine for detecting hallucinations and validating grounding

import re
from typing import Dict, Any, List, Tuple

# Common standard technologies and companies to check for unauthorized mentions
KNOWN_TECHNOLOGIES = {
    "python", "java", "javascript", "typescript", "c++", "rust", "golang", "ruby", "php",
    "html", "css", "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "spring", "docker", "kubernetes", "aws", "gcp", "azure", "sql", "nosql", "mongodb",
    "postgresql", "mysql", "redis", "elasticsearch", "spark", "hadoop", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy", "terraform", "ansible", "jenkins"
}

# Ambiguous words that require case-sensitive or contextual checks
AMBIGUOUS_TECH = {
    "go": "golang",
    "rest": "restful"
}

KNOWN_COMPANIES = {
    "google", "microsoft", "meta", "facebook", "amazon", "apple", "netflix", "uber",
    "lyft", "airbnb", "spotify", "stripe", "salesforce", "oracle", "ibm", "intel",
    "nvidia", "amd", "adobe", "twitter", "linkedin", "github", "gitlab", "redrob"
}

class GroundednessValidator:
    """Validator class to verify generated text matches provided Candidate Intelligence Package facts."""
    
    @staticmethod
    def extract_allowed_facts(cip: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extracts allowed entities, skills, numbers, and metrics from the candidate package."""
        facts = {
            "skills": [],
            "numbers": [],
            "signals": [],
            "metrics": []
        }
        
        # Extract skills and technologies from signals and JD coverage
        for sig in cip.get("positive_signals", []):
            facts["skills"].append(str(sig).lower())
        for sig in cip.get("negative_signals", []):
            facts["skills"].append(str(sig).lower())
            
        jd = cip.get("jd_coverage", {})
        for category in ["must", "preferred", "missing"]:
            for item in jd.get(category, []):
                facts["skills"].append(str(item).lower())
                
        # Extract numeric scores and metrics
        for key in ["rank", "score", "technical_fit", "career_evidence", "behavior", "integrity", "confidence"]:
            if key in cip:
                val = cip[key]
                facts["numbers"].append(str(val))
                facts["metrics"].append(key.replace("_", " "))
                
        # Extract feature contribution numbers
        fc = cip.get("feature_contributions", {})
        for k, v in fc.items():
            facts["numbers"].append(str(v))
            facts["metrics"].append(k.lower())
            
        # Clean duplicates
        facts["skills"] = list(set(facts["skills"]))
        facts["numbers"] = list(set(facts["numbers"]))
        facts["metrics"] = list(set(facts["metrics"]))
        
        return facts

    @classmethod
    def validate(cls, cip: Dict[str, Any], explanation: str) -> Tuple[bool, List[str]]:
        """Validates that explanation only contains supplied evidence.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_violations)
        """
        violations = []
        explanation_lower = explanation.lower()
        facts = cls.extract_allowed_facts(cip)
        
        # 1. Check for unauthorized technologies/skills
        words = set(re.findall(r'\b[a-zA-Z\+\#\-\.\d]+\b', explanation_lower))
        for word in words:
            # Check ambiguous tech words (like 'go') only if they are capitalized in the original text
            if word in AMBIGUOUS_TECH:
                original_instances = re.findall(rf'\b{word}\b', explanation, re.IGNORECASE)
                has_capitalized = any(inst[0].isupper() for inst in original_instances if inst)
                if not has_capitalized:
                    continue  # Skip lowercase usage as it's likely a common English verb/noun
            
            if word in KNOWN_TECHNOLOGIES:
                matched = False
                for allowed_skill in facts["skills"]:
                    if word in allowed_skill or allowed_skill in word:
                        matched = True
                        break
                if not matched:
                    violations.append(f"Unauthorized technology/skill mentioned: '{word}'")
                    
        # 2. Check for unauthorized companies
        for word in words:
            if word in KNOWN_COMPANIES:
                # Check if company was explicitly allowed
                matched = False
                for allowed_skill in facts["skills"]:
                    if word in allowed_skill:
                        matched = True
                        break
                if not matched:
                    violations.append(f"Unauthorized company mentioned: '{word}'")

        # 3. Check for invented metrics and numbers
        found_numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', explanation_lower)
        for num_str in found_numbers:
            clean_num = num_str.replace("%", "")
            matched = False
            for allowed_num in facts["numbers"]:
                if clean_num == allowed_num:
                    matched = True
                    break
                try:
                    if abs(float(clean_num) - float(allowed_num)) < 1e-4:
                        matched = True
                        break
                    if abs(float(clean_num) - float(allowed_num) * 100) < 1e-2:
                        matched = True
                        break
                except ValueError:
                    pass
            if not matched and clean_num != "100":
                violations.append(f"Inventoried or ungrounded number/metric mentioned: '{num_str}'")

        is_valid = len(violations) == 0
        return is_valid, violations
pre
