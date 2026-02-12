import pandas as pd
import json
import re
from collections import Counter
from pathlib import Path
import os
from typing import List, Dict, Tuple, Optional

# English prefix: "politically {conservative/liberal/centrist}" + stance
POLITICAL_TO_EN = {
    "Left": "politically liberal",
    "Right": "politically conservative",
    "Center": "politically centrist",
    "Undecided": "politically undecided",
}
STANCE_TO_EN = {
    "Support": "supporting",
    "Against": "opposing",
    "Neutral": "neutral",
    "Undecided": "undecided",
}


def normalize_topic(topic: str) -> str:
    """Topic for prefix: strip timestamp suffix if present (e.g. civil_liberties_20250804_181052 -> civil_liberties)."""
    s = str(topic or "").strip()
    m = re.match(r"^(.+?)_\d{8}_\d{6}$", s)
    return m.group(1) if m else (s or "the topic")


def make_prefix(topic: str, political: Optional[str], stance: Optional[str]) -> str:
    """Build English prefix: politically conservative/liberal/centrist + stance."""
    topic_label = normalize_topic(topic)
    if political and stance:
        pol_en = POLITICAL_TO_EN.get(str(political).strip(), str(political).strip() or "politically undecided")
        stance_en = STANCE_TO_EN.get(str(stance).strip(), str(stance).strip() or "undecided")
        return f"This text discusses {topic_label} from a {pol_en} perspective, {stance_en} on the issue.\n\n"
    return f"This text discusses {topic_label}.\n\n"


def get_main_perspective_from_row(row, ann_cols: List[str]) -> Optional[Tuple[str, str]]:
    """Get (political, stance) from gpt-4.1 annotation columns; majority vote across columns."""
    perspectives = []
    for col in ann_cols:
        if col not in row.index or pd.isna(row[col]) or not isinstance(row[col], str):
            continue
        try:
            ann = json.loads(row[col])
            if "Political" in ann and "Stance" in ann:
                pol = ann["Political"].get("label", "")
                stance = ann["Stance"].get("label", "")
                if pol and stance:
                    perspectives.append((pol, stance))
        except Exception:
            pass
    if not perspectives:
        return None
    counter = Counter(perspectives)
    return counter.most_common(1)[0][0]


def load_annotated_csv(csv_path: Path, ann_cols: List[str]) -> List[Dict[str, str]]:
    """
    Load original samples from annotated_dataset CSV.
    Labels from gpt-4.1 annotation columns (majority vote). Prefix: 'This text discusses {topic} from a politically conservative/liberal/centrist perspective, supporting/opposing/neutral on the issue.'
    """
    df = pd.read_csv(csv_path)
    out = []
    for idx, row in df.iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        topic = row.get("topic", "")
        persp = get_main_perspective_from_row(row, ann_cols)
        political = persp[0] if persp else None
        stance = persp[1] if persp else None
        prefix = make_prefix(topic, political, stance)
        out.append({"text": prefix + text})
    return out


def load_agent_generated_only(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load only generated rows from agents_output CSV. Labels from political, stance columns.
    Prefix: 'This text discusses {topic} from a politically conservative/liberal/centrist perspective, supporting/opposing/neutral on the issue.'
    """
    df = pd.read_csv(csv_path)
    has_content_type = "content_type" in df.columns
    has_query = "query" in df.columns
    if has_content_type:
        generated_mask = df["content_type"].notna() & (df["content_type"].astype(str).str.strip() != "")
    elif has_query:
        generated_mask = df["query"].notna() & (df["query"].astype(str).str.strip() != "")
    else:
        generated_mask = pd.Series(False, index=df.index)

    out = []
    for idx, row in df.loc[generated_mask].iterrows():
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        topic = row.get("topic", "")
        political = row.get("political")
        stance = row.get("stance")
        if pd.isna(political):
            political = None
        if pd.isna(stance):
            stance = None
        prefix = make_prefix(topic, political, stance)
        out.append({"text": prefix + text})
    return out


def save_jsonl(data: List[Dict[str, str]], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_finetuning_datasets(
    annotated_dir: str = "../../2_rag/annotated_dataset",
    output_agent_dir: str = "../../2_rag/agents_output",
    finetuning_output_dir: str = "finetuning_data",
):
    """
    Prepare fine-tuning datasets: original from annotated_dataset, agent-generated from agents_output.

    - annotated_dir: folder with annotated CSVs; labels from gpt-4.1 annotation columns (majority vote).
    - output_agent_dir: folder with *_with_generated.csv; only generated rows, labels from political/stance.
    - Prefix (English): "This text discusses {topic} from a politically conservative/liberal/centrist perspective, supporting/opposing/neutral on the issue."
    - Writes: original_imbalanced.jsonl (original only), agent_balanced.jsonl (original + agent-generated, for chunking/training).
    """
    base_dir = Path(__file__).resolve().parent
    annotated_path = (base_dir / annotated_dir).resolve()
    output_agent_path = (base_dir / output_agent_dir).resolve()
    finetuning_path = (base_dir / finetuning_output_dir).resolve()

    annotated_csvs = sorted(annotated_path.glob("*.csv"))
    agent_csvs = sorted(output_agent_path.glob("*.csv"))

    if not annotated_csvs:
        print(f"No CSV files found in: {annotated_path}")
        return
    if not agent_csvs:
        print(f"No CSV files found in: {output_agent_path}")
        return

    os.makedirs(finetuning_path, exist_ok=True)

    print("=" * 60)
    print("Fine-tuning dataset preparation")
    print("=" * 60)
    print(f"Annotated dir: {annotated_path}")
    print(f"Agent output dir: {output_agent_path}")
    print(f"Output dir: {finetuning_path}")

    # 1. Original (imbalanced) from annotated_dataset — gpt-4.1 annotation, representative label
    all_original = []
    for csv_file in annotated_csvs:
        df_sample = pd.read_csv(csv_file, nrows=1)
        ann_cols = [c for c in df_sample.columns if "gpt-4.1_" in c]
        if not ann_cols:
            print(f"  Skip (no gpt-4.1 cols): {csv_file.name}")
            continue
        print(f"  Loading original: {csv_file.name}")
        rows = load_annotated_csv(csv_file, ann_cols)
        all_original.extend(rows)
        print(f"    -> {len(rows):,} samples")

    print("\n" + "=" * 60)
    print("1. Original imbalanced")
    print("=" * 60)
    save_jsonl(all_original, str(finetuning_path / "original_imbalanced.jsonl"))
    print(f"  Saved: {len(all_original):,} samples -> {finetuning_path / 'original_imbalanced.jsonl'}")

    # 2. Agent-generated rows (for combined)
    all_generated = []
    for csv_file in agent_csvs:
        print(f"  Loading agent-generated: {csv_file.name}")
        rows = load_agent_generated_only(csv_file)
        all_generated.extend(rows)
        print(f"    -> {len(rows):,} samples")

    # 3. Agent-balanced = combined (original + agent-generated), for chunking/training
    print("\n" + "=" * 60)
    print("2. Agent balanced (original + agent-generated)")
    print("=" * 60)
    all_combined = all_original + all_generated
    save_jsonl(all_combined, str(finetuning_path / "agent_balanced.jsonl"))
    print(f"  Saved: {len(all_combined):,} samples -> {finetuning_path / 'agent_balanced.jsonl'}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Original (imbalanced): {len(all_original):,} -> original_imbalanced.jsonl")
    print(f"  Agent balanced (combined): {len(all_combined):,} -> agent_balanced.jsonl")

    if all_original:
        print(f"\n  Sample (original): {all_original[0]['text'][:120]}...")
    if all_combined:
        print(f"  Sample (agent_balanced): {all_combined[0]['text'][:120]}...")


if __name__ == "__main__":
    prepare_finetuning_datasets()
