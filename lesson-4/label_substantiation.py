from __future__ import annotations

"""Generate ground-truth labels for the *unsupported / unsubstantiated information*
failure mode in NurtureBoss conversations.

For every record in ``lesson-4/nurtureboss_traces.json`` this script issues a
single LLM call (OpenAI **gpt-4o-mini**) asking: *Are all agent responses fully
substantiated?*  The model must output strict JSON:

    {"all_responses_substantiated": true/false, "rationale": "<≤2 sentences>"}

The boolean is attached back onto the record under the same key and the updated
list is saved as ``lesson-4/nurtureboss_traces_labeled.json``.

The script is incremental: records already containing
"all_responses_substantiated" are skipped, so you can resume labeling later.
Run from repo root:

    python lesson-4/label_substantiation.py
"""

from pathlib import Path
import hashlib
import json
import os
import sys
from collections import Counter
from typing import List, Dict, Any

import litellm  # type: ignore
from concurrent.futures import ThreadPoolExecutor, as_completed
# tqdm is optional – fall back to identity iterator if missing
try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("lesson-4/nurtureboss_traces.json")
OUTPUT_PATH = Path("lesson-4/nurtureboss_traces_labeled.json")
MODEL_NAME = "gpt-4.1"

# Thread-pool size; override via env variable if desired
MAX_WORKERS = int(os.environ.get("NB_LLM_WORKERS", "64"))


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------


class SubstantiationResult(BaseModel):
    rationale: str
    all_responses_substantiated: bool


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def stable_hash(text: str, length: int = 8) -> str:
    """Return a short, deterministic hash of *text* for caching keys."""
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def build_prompt(messages: List[Dict[str, str]], metadata: Dict[str, Any] | None, z_note: str | None) -> str:
    """Craft the evaluation prompt with few-shot examples and task definition."""
    # Flatten conversation for display
    conversation_lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_lines.append(f"{role.upper()}: {content}")
    conversation_block = "\n".join(conversation_lines)

    metadata_block = json.dumps(metadata, indent=2, ensure_ascii=False) if metadata else "<no metadata>"
    z_note_block = z_note or "<no z_note>"

    prompt = f"""
You are a quality-control judge for AI leasing assistants.

Failure mode to detect: *Unsupported or unsubstantiated information* – any time the assistant states a fact that cannot be verified by the sources listed below.

The ***Tool / Metadata*** section presents either (1) direct outputs returned by internal tools (for example, `get_unit_availability`) **or** (2) the exact arguments with which those tools could be invoked. If a factual claim would be returned by calling one of these tools with arguments that are plainly inferable from the conversation (e.g., the user asks about apartment *A11* and a tool can be called with `{{"unit":"A11"}}`), then that claim **is substantiated**.

Evaluation rubric
PASS ⇒ Every assistant statement is supported by at least one of:
    • Information already provided by the user,  
    • The Tool / Metadata section, **or**  
    • A tool that could be called using arguments directly derivable from the conversation.
FAIL ⇒ At least one assistant statement lacks such support.

Focus strictly on the original user query, the assistant replies, and the Tool / Metadata block. Ignore politeness or generic filler.
If the user asks about something that is not explicitly mentioned in the tool args and tool response, do not count it as substantiated.

Evaluate the **actual conversation** below.
Return ONLY JSON with the two keys shown above – no additional text.

=== Conversation ===
{conversation_block}
=== End Conversation ===

=== Tool / Metadata (if any) ===
{metadata_block}
=== End Tool / Metadata ===

=== Reviewer Note (if any) ===
{z_note_block}
=== End Reviewer Note ===
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Main labeling routine
# ---------------------------------------------------------------------------

def main() -> None:
    with DATA_PATH.open() as fp:
        records: List[Dict[str, Any]] = json.load(fp)

    # Prepare work list -------------------------------------------------------
    pending: List[Dict[str, Any]] = [r for r in records if "all_responses_substantiated" not in r]

    def evaluate(rec: Dict[str, Any]) -> SubstantiationResult:
        """Return (id, llm JSON output string)."""
        prompt = build_prompt(
            messages=rec.get("messages", []),
            metadata={k: v for k, v in rec.items() if k not in {"messages", "id", "z_note"}},
            z_note=rec.get("z_note"),
        )

        resp = litellm.completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format=SubstantiationResult,
            temperature=0,
        )
        resp = SubstantiationResult(**json.loads(resp.choices[0].message.content))
        return resp

    updated = 0
    # Thread-pool execution ----------------------------------------------------
    if pending:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(evaluate, r): r for r in pending}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
                rec = futures[fut]
                parsed = fut.result()
                rec["all_responses_substantiated"] = parsed.all_responses_substantiated
                rec["substantiation_rationale"] = parsed.rationale
                updated += 1

    if updated:
        with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
            json.dump(records, fp, indent=2, ensure_ascii=False)
        print(f"Wrote {updated} new labels → {OUTPUT_PATH}")
    else:
        print("Nothing to label.")
        
    # Print distribution of labels
    labels = [r["all_responses_substantiated"] for r in records]
    print(f"Label distribution: {Counter(labels)}")


if __name__ == "__main__":
    main() 