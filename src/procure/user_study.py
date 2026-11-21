"""User-study pack: scenario sheets, SUS and trust/understanding questionnaires, scoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from procure.config import _read_yaml, path, resolve


def sus_score(responses: list[int]) -> float:
    """System Usability Scale score (0-100) from 10 answers on 1-5.
    Odd items: answer - 1; even items: 5 - answer; sum x 2.5 (Brooke, 1996)."""
    if len(responses) != 10 or not all(1 <= r <= 5 for r in responses):
        raise ValueError("SUS needs 10 answers between 1 and 5")
    return 2.5 * sum((r - 1) if i % 2 == 0 else (5 - r) for i, r in enumerate(responses))


def score_trust(responses: dict[str, list[int]]) -> dict[str, float]:
    """Mean Likert score per trust/understanding dimension."""
    return {k: sum(v) / len(v) for k, v in responses.items()}


def write_pack(out_dir: Path | None = None) -> Path:
    """Write scenarios.md, questionnaires (markdown + CSV response templates) and instructions."""
    cfg = _read_yaml(resolve("config/user_study.yaml"))
    out = (out_dir or path("results_dir")) / "user_study"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# User-study scenarios", "", f"Participants: {cfg['participants']}.", "",
             "Run each scenario in the dashboard (make dashboard), record a decision, then answer the questionnaires.", ""]
    for s in cfg["scenarios"]:
        r = s["request"]
        lines += [f"## {s['id']}. {s['title']}", "",
                  f"- Item: {r['item']} ({r['item_category']}), quantity {r['quantity']:,}",
                  f"- Deliver to: {r['delivery_lga']}, {r['delivery_state']} State",
                  f"- Method: {r['method'].replace('_', ' ')}; budget N{r['budget']:,}; date {r['date']}",
                  f"- **Task:** {s['task']}", ""]
    (out / "scenarios.md").write_text("\n".join(lines))
    pd.DataFrame([{"scenario_id": s["id"], **s["request"]} for s in cfg["scenarios"]]).to_csv(out / "scenarios.csv", index=False)
    scale = " | ".join(cfg["likert"])
    sus = ["# System Usability Scale (SUS)", "", f"Scale: {scale}", ""] + \
          [f"{i}. {q}  [ 1  2  3  4  5 ]" for i, q in enumerate(cfg["sus_items"], 1)] + \
          ["", "Scoring: odd items (answer - 1), even items (5 - answer); sum x 2.5 = 0-100. 68 is average."]
    (out / "sus_questionnaire.md").write_text("\n".join(sus) + "\n")
    trust = ["# Trust and understanding questionnaire", "", f"Scale: {scale}", ""]
    n = 0
    for dim, qs in cfg["trust_items"].items():
        trust += [f"## {dim.capitalize()}", ""]
        for q in qs:
            n += 1
            trust.append(f"{n}. {q}  [ 1  2  3  4  5 ]")
        trust.append("")
    trust += ["## Open questions", "", "a. Which explanation was most useful, and why?",
              "b. Was there a recommendation you disagreed with? What did you do?",
              "c. What would stop you from using this system at work?"]
    (out / "trust_questionnaire.md").write_text("\n".join(trust) + "\n")
    cols = ["participant_id", "scenario_id"] + [f"sus_{i}" for i in range(1, 11)] + \
           [f"{d}_{i}" for d, qs in cfg["trust_items"].items() for i in range(1, len(qs) + 1)]
    pd.DataFrame(columns=cols).to_csv(out / "responses_template.csv", index=False)
    (out / "procedure.md").write_text(
        "# Procedure\n\n1. Obtain informed consent (record participant IDs only; no names - NDPA 2023).\n"
        "2. 10-minute dashboard demonstration using a practice request.\n"
        "3. Participants complete scenarios S1-S5 in random order; decisions are captured in the audit log.\n"
        "4. Participants complete the SUS and the trust/understanding questionnaire.\n"
        "5. Score with procure.user_study.sus_score / score_trust; report mean, SD and median per item.\n")
    return out
