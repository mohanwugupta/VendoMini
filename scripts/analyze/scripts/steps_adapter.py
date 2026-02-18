import json
from typing import Dict, Any, Iterable, List, Optional

def iter_steps_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Iterate over lines in a steps.jsonl file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def extract_assistant_text(step: Dict[str, Any]) -> str:
    """
    Extract assistant text from a step.
    Your JSONL has no explicit 'assistant message' stream.
    The closest proxy is whatever the model wrote while predicting/planning.
    """
    # 1. Prediction object (primary source of model output in VendoMini)
    pred = step.get("prediction", {}) or {}
    
    # Try scratchpad_raw first as it contains the code/thought trace
    if isinstance(pred, dict):
        raw = pred.get("scratchpad_raw")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()

        # Try explicit prediction text
        txt = pred.get("prediction_text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
            
    # Fallback: check for top-level scratchpad_raw (some logs structure it flat)
    raw_flat = step.get("scratchpad_raw")
    if isinstance(raw_flat, str) and raw_flat.strip():
        return raw_flat.strip()

    # 2. Action thought/reasoning (legacy/alternative format)
    action = step.get("action", {})
    if isinstance(action, dict):
        reasoning = action.get("reasoning") or action.get("thought")
        if reasoning and isinstance(reasoning, str):
            return reasoning.strip()
        
        # If write_scratchpad tool, the content is the text
        if action.get("tool") in ["tool_write_scratchpad", "write_scratchpad"]:
             content = action.get("content") or action.get("text")
             if content and isinstance(content, str):
                 return content.strip()

    return ""
