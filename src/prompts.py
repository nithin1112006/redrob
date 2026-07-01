# Prompt templates and chat structure utilities

import json
from typing import Dict, Any

# System message enforcing factual grounding and style constraints
SYSTEM_PROMPT_TEMPLATE = (
    "You are an experienced senior recruiter.\n"
    "Never invent facts.\n"
    "Never hallucinate.\n"
    "Only explain supplied evidence.\n"
    "Write the explanation in a {style} style.\n"
    "Adhere strictly to the provided Candidate Intelligence Package (CIP) facts."
)

def get_system_prompt(style: str = "Recruiter") -> str:
    """Returns the system prompt tailored to the requested explanation style."""
    valid_styles = ["Professional", "Recruiter", "Concise", "Technical"]
    if style not in valid_styles:
        style = "Recruiter"
    return SYSTEM_PROMPT_TEMPLATE.format(style=style)

def format_user_prompt(cip: Dict[str, Any]) -> str:
    """Formats the Candidate Intelligence Package dictionary as a JSON string for the user message."""
    return json.dumps(cip, indent=2)

def create_messages(cip: Dict[str, Any], style: str = "Recruiter", response: str = None) -> list:
    """Creates a Hugging Face message list representing the dialog interaction."""
    messages = [
        {"role": "system", "content": get_system_prompt(style)},
        {"role": "user", "content": format_user_prompt(cip)}
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
