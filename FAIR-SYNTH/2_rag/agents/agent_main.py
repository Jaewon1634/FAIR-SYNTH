import os
import sys
import json
import time
from collections import defaultdict

# ensure agents dir is on path for multi_agents
_AGENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENTS_DIR not in sys.path:
    sys.path.insert(0, _AGENTS_DIR)

import pandas as pd
from tqdm import tqdm

from multi_agents import (
    RAGPipeline,
    PipelineState,
    ANNOTATED_DATASET_DIR,
    OUTPUT_DIR,
    AGENTS_OUTPUT_DIR,
)


def run_all_topics_sequential():
    start_total = time.time()

    print("=" * 80)
    print("RUN ALL TOPICS SEQUENTIAL (with progress)")
    print("=" * 80)

    pipeline = RAGPipeline()

    # 1) Load dataset
    state: PipelineState = PipelineState()
    state.update(pipeline.load_dataset_node(state))

    # 2) Analyze distribution
    state.update(pipeline.analyze_distribution_node(state))

    deficits = state["deficits"]
    topic_base_map = state.get("topic_base_map", {})

    # Group deficits by topic (order: same as state["topics"])
    topics_order = state["topics"]
    deficits_by_topic = defaultdict(list)
    for d in deficits:
        deficits_by_topic[d["topic"]].append(dict(d))

    total_generations = sum(d["deficit"] for d in deficits)
    print(f"\nTotal topics: {len(topics_order)}")
    print(f"Total deficits: {len(deficits)}")
    print(f"Total generations to run: {total_generations}")

    # Shared state across the run
    generation_results = []
    content_type_counter = defaultdict(int)
    angle_counter = defaultdict(int)
    total_generated = 0

    # One progress bar for total generations (with ETA)
    pbar_total = tqdm(
        total=total_generations,
        desc="Total generations",
        unit="gen",
        dynamic_ncols=True,
        leave=True,
    )

    for topic_full in tqdm(topics_order, desc="Topics", unit="topic", leave=True):
        topic_deficits = deficits_by_topic.get(topic_full, [])
        if not topic_deficits:
            continue

        topic_start = time.time()
        topic_gen_count = 0

        for deficit_item in topic_deficits:
            needed = deficit_item["deficit"]
            political = deficit_item["political"]
            stance = deficit_item["stance"]

            # Local state for this deficit (single deficit, multiple iterations)
            local_state: PipelineState = PipelineState()
            local_state.update({
                "dataset": state["dataset"],
                "topics": state["topics"],
                "topic_base_map": topic_base_map,
                "deficits": [deficit_item],
                "current_deficit_idx": 0,
                "generation_results": [],
                "content_type_counter": defaultdict(int),
                "style_counter": defaultdict(int),
                "angle_counter": defaultdict(int),
            })

            for iteration in range(needed):
                local_state["current_iteration"] = iteration

                # Node 3: search
                s3 = pipeline.search_documents_node(local_state)
                local_state.update(s3)

                if not local_state.get("sampled_originals"):
                    pbar_total.update(1)
                    continue

                # Node 4: outline
                s4 = pipeline.outline_generation_node(local_state)
                local_state.update(s4)

                if not local_state.get("outline"):
                    pbar_total.update(1)
                    continue

                # Node 5: content
                s5 = pipeline.content_generation_node(local_state)
                local_state.update(s5)

                if not local_state.get("generated_text"):
                    pbar_total.update(1)
                    continue

                # Save result (same structure as save_result_node)
                outline = local_state["outline"]
                result = {
                    "topic": local_state["current_topic"],
                    "political": local_state["current_political"],
                    "stance": local_state["current_stance"],
                    "text": local_state["generated_text"],
                    "query": local_state.get("current_query", ""),
                    "retrieved_contexts": list(local_state.get("sampled_originals", [])),
                    "content_type": outline.get("content_type", ""),
                    "title": outline.get("title", ""),
                    "angle": outline.get("angle", ""),
                    "target_audience": outline.get("target_audience", ""),
                    "key_points": outline.get("key_points", []),
                    "reasoning": outline.get("reasoning", ""),
                    "num_context_docs": len(local_state.get("sampled_originals", [])),
                }
                generation_results.append(result)
                content_type_counter[outline.get("content_type", "")] += 1
                angle_counter[outline.get("angle", "")] += 1
                total_generated += 1
                topic_gen_count += 1

                pbar_total.update(1)
                pbar_total.set_postfix(topic=topic_full[:20], total=total_generated, refresh=True)

        topic_elapsed = time.time() - topic_start
        tqdm.write(f"  Topic {topic_full}: {topic_gen_count} generated in {topic_elapsed:.1f}s")

        # Save per topic as soon as this topic is done (so partial progress is persisted)
        topic_results = [r for r in generation_results if r["topic"] == topic_full]
        if topic_results:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            os.makedirs(AGENTS_OUTPUT_DIR, exist_ok=True)
            output_file = os.path.join(OUTPUT_DIR, f"generated_{topic_full}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(topic_results, f, ensure_ascii=False, indent=2)
            tqdm.write(f"  Saved {len(topic_results)} results to: {output_file}")
            original_path = os.path.join(ANNOTATED_DATASET_DIR, f"annotated_{topic_full}.csv")
            if os.path.exists(original_path):
                try:
                    original_df = pd.read_csv(original_path)
                except Exception as e:
                    tqdm.write(f"  WARNING: Failed to load original for '{topic_full}': {e}")
                    original_df = pd.DataFrame()
            else:
                original_df = pd.DataFrame()
            generated_df = pd.DataFrame(topic_results)
            merged_df = pd.concat([original_df, generated_df], ignore_index=True, sort=False) if not original_df.empty else generated_df
            merged_path = os.path.join(AGENTS_OUTPUT_DIR, f"{topic_full}_with_generated.csv")
            try:
                merged_df.to_csv(merged_path, index=False)
                tqdm.write(f"  Saved merged CSV to: {merged_path}")
            except Exception as e:
                tqdm.write(f"  WARNING: Failed to save merged CSV for '{topic_full}': {e}")

    pbar_total.close()

    elapsed_total = time.time() - start_total
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Total generated: {total_generated}")
    print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    if total_generated:
        print(f"Avg per generation: {elapsed_total/total_generated:.1f}s")
    print("\nContent type distribution:")
    for ctype, count in sorted(content_type_counter.items()):
        print(f"  {ctype}: {count}")
    print("\nAngle distribution:")
    for angle, count in sorted(angle_counter.items()):
        print(f"  {angle}: {count}")

    # Results are already saved per topic when each topic completes (see loop above).
    print(f"\nTotal results in memory: {len(generation_results)}")
    return total_generated


if __name__ == "__main__":
    run_all_topics_sequential()
