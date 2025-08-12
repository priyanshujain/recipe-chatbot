"""LLM Judge for the *unsupported / unsubstantiated information* failure mode.

Reads the ground-truth file produced by *label_substantiation.py* and evaluates
a cheaper model (OpenAI **gpt-4.1-nano**) against it.

Metrics reported:
• TPR – True-Positive rate (correctly identifies Pass / failure absent).
• TNR – True-Negative rate (correctly identifies Fail / failure present).

A deterministic 10 %/10 %/80 % split (train/dev/test) is created based on a
hash of the `id` so results are repeatable.
"""

from pathlib import Path
import hashlib
import json
import os
from typing import List, Dict, Any

import litellm  # type: ignore
from sklearn.metrics import confusion_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from pydantic import BaseModel

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

load_dotenv()

DATA_PATH = Path("lesson-4/nurtureboss_traces_labeled.json")
MODEL_NAME = "gpt-4.1"

# Threadpool size for model calls
MAX_WORKERS = int(os.environ.get("NB_LLM_WORKERS", "64"))

# Desired absolute sizes
TRAIN_SIZE = 20
DEV_SIZE = 30


# ---------------------------------------------------------------------------
# Pydantic schema for judge output
# ---------------------------------------------------------------------------


class JudgeResult(BaseModel):
    all_responses_substantiated: bool
    reason: str


# ---------------------------------------------------------------------------
# Utilities
    # ---------------------------------------------------------------------------


def split_dataset(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Return fixed-size train/dev/test buckets (20/20/remaining)."""
    # Keep deterministic order by id hash
    sorted_records = sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())

    train = sorted_records[:TRAIN_SIZE]
    dev = sorted_records[TRAIN_SIZE : TRAIN_SIZE + DEV_SIZE]
    test = sorted_records[TRAIN_SIZE + DEV_SIZE :]
    return {"train": train, "dev": dev, "test": test}


# Few-shot examples injected later

def _format_conv_short(rec: Dict[str, Any]) -> str:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in rec.get("messages", [])]
    return "\n".join(lines)


def build_judge_prompt(messages: List[Dict[str, str]], metadata: Dict[str, Any] | None, *, example_pass: str, example_fail: str) -> str:
    lines = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    conv_block = "\n".join(lines)
    metadata_block = json.dumps(metadata, indent=2, ensure_ascii=False) if metadata else "<no metadata>"

    prompt = f"""
You are a quality-control judge for AI leasing assistants.

Failure mode to detect: *Unsupported or unsubstantiated information* – any time the assistant states a 
fact that cannot be verified by the provided tool outputs, prior messages, or metadata.

Decision rules (apply in order):
1. Courtesy / process statements (greetings, confirmations, transfers) NEVER require evidence – consider them substantiated.
2. If a statement cites, paraphrases, or clearly re-states information present in tool outputs or metadata, that counts as substantiated.
3. Only mark `all_responses_substantiated` as false when you can point to a *specific factual claim* that is not supported or that contradicts the provided evidence.
4. When uncertain, choose **true** (assistant is substantiated).
5. Descriptive qualifiers (e.g., "garden-style", "luxury", "in-person") or categorical claims about the property MUST appear verbatim in tool outputs/metadata to be considered substantiated – do not infer them from loosely related terms.

True  ⇒ every assistant message is fully substantiated.
False ⇒ at least one assistant message contains unsupported or contradictory information.

Few-shot examples (for calibration only – do NOT reference them in your answer):
--- Example PASS ---
{example_pass}
Expected JSON: {{"all_responses_substantiated": true, "reason": "assistant statements fully supported."}}
--- Example FAIL ---
{example_fail}
Expected JSON: {{"all_responses_substantiated": false, "reason": "assistant hallucinated."}}
--- End Examples ---

Evaluate the **actual conversation** below.
Return ONLY JSON with the two keys shown above – no additional text.

=== Conversation ===
{conv_block}
=== End Conversation ===

=== Tool / Metadata (Where any claims could be substantiated) ===
{metadata_block}
=== End Tool / Metadata ===
"""
    return prompt.strip()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_tpr_tnr(y_true: List[bool], y_pred: List[bool]):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return tpr, tnr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Missing OPENAI_API_KEY")

    with DATA_PATH.open() as fp:
        records = json.load(fp)

    splits = split_dataset(records)

    for split_name, split_recs in splits.items():
        y_true: List[bool] = [bool(r["all_responses_substantiated"]) for r in split_recs]

        # Collect pass/fail examples – ensure equal count (up to 2 each)
        pass_examples = [r for r in splits["train"] if r["all_responses_substantiated"]]
        fail_examples = [r for r in splits["train"] if not r["all_responses_substantiated"]]

        k = min(5, len(pass_examples), len(fail_examples))
        pass_examples = pass_examples[:k]
        fail_examples = fail_examples[:k]

        example_pass_combined = "\n---\n".join(_format_conv_short(r) for r in pass_examples)
        example_fail_combined = "\n---\n".join(_format_conv_short(r) for r in fail_examples)

        def evaluate(rec: Dict[str, Any]) -> bool:
            prompt = build_judge_prompt(
                rec.get("messages", []),
                {k: v for k, v in rec.items() if k not in {"messages", "id", "z_note", "all_responses_substantiated", "substantiation_rationale"}},
                example_pass=example_pass_combined,
                example_fail=example_fail_combined,
            )
            resp_raw = litellm.completion(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format=JudgeResult,
            )

            # LiteLLM still returns OpenAI-style object; extract JSON and parse
            parsed = JudgeResult(**json.loads(resp_raw.choices[0].message.content))  # type: ignore[attr-defined]
            return parsed.all_responses_substantiated

        y_pred: List[bool] = [False] * len(split_recs)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(evaluate, rec): idx for idx, rec in enumerate(split_recs)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} judge"):
                idx = futures[fut]
                y_pred[idx] = fut.result()

        tpr, tnr = compute_tpr_tnr(y_true, y_pred)
        print(f"{split_name.upper():5} — size {len(split_recs):3d} | TPR: {tpr:.2%} | TNR: {tnr:.2%}")

        # -------------------------------------------------------------------
        # Surface false positives for analysis (predicted Pass but ground Fail)
        # -------------------------------------------------------------------
        false_positives = [
            rec for rec, pred_flag in zip(split_recs, y_pred)
            if pred_flag and not rec["all_responses_substantiated"]
        ]

        if false_positives:
            print(f"--- FALSE POSITIVES ({len(false_positives)}) in {split_name} ---")
            for fp in false_positives:
                print(f"ID: {fp['id']}")
                rationale = fp.get('substantiation_rationale', '<no rationale>')
                print(f"Rationale: {rationale}\n")


if __name__ == "__main__":
    main() 