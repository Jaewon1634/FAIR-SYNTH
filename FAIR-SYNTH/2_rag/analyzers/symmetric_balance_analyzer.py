import chromadb
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import json

# Configuration
CHROMA_DB_PATH = "chroma_db"  # Created within 2_rag folder
COLLECTION_NAME = "annotated_docs"

# Symmetric pair definitions
SYMMETRIC_PAIRS = [
    ('Left-Against', 'Right-Support'),
    ('Left-Support', 'Right-Against'),
    ('Left-Neutral', 'Right-Neutral'),
    ('Center-Against', 'Center-Support'),
]

# Standalone combinations (no balance needed)
STANDALONE = ['Center-Neutral']

# Excluded combinations
EXCLUDED = ['Undecided-Undecided', 'Undecided-Neutral']


def analyze_topic_distribution(collection, topic_name: str) -> Dict[str, int]:
    """Calculate distribution by combination for a specific topic"""
    # Fetch documents for the topic (metadata only)
    results = collection.get(
        where={"topic": {"$eq": topic_name}},
        include=['metadatas']
    )
    
    if not results['metadatas']:
        return {}
    
    # Count by combination (deduplicate by original_row_id)
    seen_rows = set()
    combination_counts = Counter()
    
    for meta in results['metadatas']:
        row_id = meta.get('original_row_id')
        if row_id in seen_rows:
            continue
        seen_rows.add(row_id)
        
        political = meta.get('political_major', 'Undecided')
        stance = meta.get('stance_major', 'Undecided')
        combo = f"{political}-{stance}"
        combination_counts[combo] += 1
    
    return dict(combination_counts)


def calculate_symmetric_deficits(distribution: Dict[str, int]) -> List[Dict]:
    """Calculate deficit based on symmetric pairs"""
    deficits = []
    
    for combo1, combo2 in SYMMETRIC_PAIRS:
        count1 = distribution.get(combo1, 0)
        count2 = distribution.get(combo2, 0)
        
        # Set the larger count as target
        if count1 > count2:
            target_count = count1
            underrepresented = combo2
            deficit = count1 - count2
        elif count2 > count1:
            target_count = count2
            underrepresented = combo1
            deficit = count2 - count1
        else:
            # Balanced if equal
            continue
        
        deficits.append({
            'pair': (combo1, combo2),
            'underrepresented': underrepresented,
            'current_count': min(count1, count2),
            'target_count': target_count,
            'deficit': deficit
        })
    
    return deficits


def analyze_all_topics(collection) -> Dict[str, Dict]:
    """Perform balance analysis for all topics"""
    # Get all topic names
    all_metas = collection.get(include=['metadatas'])
    topics = set(meta.get('topic') for meta in all_metas['metadatas'] if meta.get('topic'))
    
    print(f"Topics to analyze: {len(topics)}")
    
    results = {}
    
    for topic in sorted(topics):
        print(f"\n{'='*100}")
        print(f"Topic: {topic}")
        print(f"{'='*100}")
        
        # Calculate distribution
        distribution = analyze_topic_distribution(collection, topic)
        
        # Total sample count
        total_samples = sum(distribution.values())
        print(f"Total samples: {total_samples:,}")
        
        # Print distribution by combination
        print(f"\nDistribution by combination:")
        for combo, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            print(f"  {combo:25s}: {count:5,} ({percentage:5.2f}%)")
        
        # Analyze symmetric pairs
        deficits = calculate_symmetric_deficits(distribution)
        
        if deficits:
            print(f"\nUnderrepresented combinations (symmetric pair-based):")
            for item in deficits:
                pair_str = f"{item['pair'][0]} <-> {item['pair'][1]}"
                print(f"  {pair_str:50s}")
                print(f"    -> {item['underrepresented']:25s}: {item['current_count']:5,} -> {item['target_count']:5,} (deficit: {item['deficit']:5,})")
        else:
            print(f"\nAll symmetric pairs are balanced")
        
        results[topic] = {
            'distribution': distribution,
            'total_samples': total_samples,
            'deficits': deficits
        }
    
    return results


def save_analysis_results(results: Dict, output_file: str = "symmetric_balance_analysis.json"):
    """Save analysis results"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis results saved: {output_file}")


def main():
    print("=" * 100)
    print("Symmetric Pair-based Balance Analysis Started")
    print("=" * 100)
    
    # Connect to ChromaDB
    print(f"\nConnecting to ChromaDB: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Get collection
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' loaded (document count: {collection.count():,})")
    
    # Print symmetric pair definitions
    print(f"\nSymmetric pair definitions:")
    for combo1, combo2 in SYMMETRIC_PAIRS:
        print(f"  {combo1:25s} <-> {combo2:25s}")
    print(f"\nStandalone combinations: {', '.join(STANDALONE)}")
    print(f"Excluded combinations: {', '.join(EXCLUDED)}")
    
    # Analyze all topics
    results = analyze_all_topics(collection)
    
    # Save results
    save_analysis_results(results)
    
    # Overall summary
    print("\n" + "=" * 100)
    print("Overall Summary")
    print("=" * 100)
    
    total_deficits = 0
    for topic, data in results.items():
        topic_deficits = sum(d['deficit'] for d in data['deficits'])
        total_deficits += topic_deficits
        if topic_deficits > 0:
            print(f"  {topic:20s}: {topic_deficits:6,} samples needed")
    
    print(f"\nTotal samples to generate: {total_deficits:,}")
    
    print("\n" + "=" * 100)
    print("Analysis completed")
    print("=" * 100)


if __name__ == "__main__":
    main()
