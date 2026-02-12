import os
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from collections import Counter
from typing import Tuple, List

# Configuration
ANNOTATED_DIR = "/workspace/rag_fair_dataset_SIGIR2026/2_rag/annotated_dataset"
CHROMA_DB_PATH = "/workspace/rag_fair_dataset_SIGIR2026/1_vectordb/C4_dataset/chroma_db"
COLLECTION_NAME = "c4_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

# Chunking settings (token-based)
MAX_TOKENS = 512  # Maximum sequence length for multilingual-e5-large-instruct
OVERLAP_TOKENS = 50  # Overlap tokens
BATCH_SIZE = 500


def extract_labels_and_scores(json_str: str) -> Tuple[str, str, float, float]:
    """Extract labels and scores from GPT annotation"""
    try:
        data = json.loads(json_str)
        political_label = data.get('Political', {}).get('label', 'Undecided')
        stance_label = data.get('Stance', {}).get('label', 'Undecided')
        political_score = data.get('Political', {}).get('score', 0.0)
        stance_score = data.get('Stance', {}).get('score', 0.0)
        return political_label, stance_label, political_score, stance_score
    except:
        return 'Undecided', 'Undecided', 0.0, 0.0


def chunk_text_by_tokens(text: str, tokenizer, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> List[str]:
    """Chunk text by tokens (multilingual-e5-large-instruct has 512 token limit)"""
    # Tokenize entire text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) <= max_tokens:
        decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
        # Verify after decode-encode cycle
        reencoded = tokenizer.encode(decoded_text, add_special_tokens=False)
        if len(reencoded) <= max_tokens:
            return [decoded_text.strip()]
        else:
            # Truncate if re-encoding exceeds limit
            truncated = tokens[:max_tokens]
            return [tokenizer.decode(truncated, skip_special_tokens=True).strip()]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Use safety margin (510 tokens instead of 512) to account for decode-encode differences
        # This is NOT content loss - it's just a safety buffer to prevent exceeding 512 after re-encoding
        safe_max = max_tokens - 2
        end_idx = min(start_idx + safe_max, len(tokens))
        
        # Convert tokens back to text
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Verify token count after decode-encode cycle
        # Sometimes decode-encode can add 1-2 tokens, so we check
        reencoded = tokenizer.encode(chunk_text, add_special_tokens=False)
        if len(reencoded) > max_tokens:
            # Rare case: even with safety margin, re-encoding exceeds limit
            # Try to find the maximum size that fits (may cause minimal content loss)
            for test_end in range(end_idx, start_idx, -1):
                test_tokens = tokens[start_idx:test_end]
                test_text = tokenizer.decode(test_tokens, skip_special_tokens=True)
                test_reencoded = tokenizer.encode(test_text, add_special_tokens=False)
                if len(test_reencoded) <= max_tokens:
                    chunk_text = test_text
                    end_idx = test_end
                    break
            else:
                # Final fallback: truncate re-encoded tokens (minimal content loss)
                final_tokens = reencoded[:max_tokens]
                chunk_text = tokenizer.decode(final_tokens, skip_special_tokens=True)
        
        chunks.append(chunk_text.strip())
        
        # Set next start point for overlap
        if end_idx >= len(tokens):
            break
        start_idx = end_idx - overlap_tokens
    
    return chunks


def determine_major_labels(annotations: List[Tuple[str, str, float, float]]) -> Tuple[str, str, float, float]:
    """Determine major political/stance labels through majority voting"""
    political_votes = [ann[0] for ann in annotations]
    stance_votes = [ann[1] for ann in annotations]
    political_scores = [ann[2] for ann in annotations]
    stance_scores = [ann[3] for ann in annotations]
    
    avg_political_score = sum(political_scores) / len(political_scores) if political_scores else 0.0
    avg_stance_score = sum(stance_scores) / len(stance_scores) if stance_scores else 0.0
    
    # Political majority voting
    political_counter = Counter(political_votes)
    political_most_common = political_counter.most_common(2)
    
    if len(political_most_common) == 1:
        major_political = political_most_common[0][0]
    elif political_most_common[0][1] > political_most_common[1][1]:
        major_political = political_most_common[0][0]
    else:
        if -0.2 <= avg_political_score <= 0.2:
            major_political = 'Center'
        elif avg_political_score < 0:
            major_political = 'Left'
        else:
            major_political = 'Right'
    
    # Stance majority voting
    stance_counter = Counter(stance_votes)
    stance_most_common = stance_counter.most_common(2)
    
    if len(stance_most_common) == 1:
        major_stance = stance_most_common[0][0]
    elif stance_most_common[0][1] > stance_most_common[1][1]:
        major_stance = stance_most_common[0][0]
    else:
        if -0.2 <= avg_stance_score <= 0.2:
            major_stance = 'Neutral'
        elif avg_stance_score < 0:
            major_stance = 'Against'
        else:
            major_stance = 'Support'
    
    return major_political, major_stance, avg_political_score, avg_stance_score


def process_csv_file(csv_file: str, collection, tokenizer):
    """Process CSV file: major voting + token-based chunking + vector DB storage"""
    csv_path = os.path.join(ANNOTATED_DIR, csv_file)
    df = pd.read_csv(csv_path)
    
    print(f"\nProcessing: {csv_file} ({len(df)} rows)")
    
    gpt_cols = [
        'gpt-4.1_opp_left',
        'gpt-4.1_opp_right',
        'gpt-4.1_sup_left', 
        'gpt-4.1_sup_right'
    ]
    
    all_chunks = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Chunking"):
        text = str(row.get('text', '')).strip()
        topic = str(row.get('topic', '')).strip()
        url = str(row.get('url', ''))
        
        if not text or len(text) < 50:
            continue
        
        # Collect 4 persona annotations
        annotations = []
        for col in gpt_cols:
            pol_label, stance_label, pol_score, stance_score = extract_labels_and_scores(row[col])
            annotations.append((pol_label, stance_label, pol_score, stance_score))
        
        # Determine representative label through major voting
        major_political, major_stance, avg_pol_score, avg_stance_score = determine_major_labels(annotations)
        
        # Token-based text chunking (512 token limit)
        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS)
        
        for chunk_id, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text or len(chunk_text) < 50:
                continue
            
            # Verify token count (should be <= 512)
            tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
            token_count = len(tokens)
            
            # Warning if exceeds 512 tokens (should not happen theoretically)
            if token_count > MAX_TOKENS:
                print(f"  WARNING: Chunk {chunk_id} exceeds limit with {token_count} tokens")
                # Force truncation
                chunk_tokens = tokens[:MAX_TOKENS]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                token_count = MAX_TOKENS
            
            doc_id = f"{csv_file}_{idx}_chunk_{chunk_id}"
            
            all_chunks.append({
                "id": doc_id,
                "text": chunk_text,
                "metadata": {
                    "topic": topic,
                    "political_major": major_political,
                    "stance_major": major_stance,
                    "original_text": text,  # Preserve full original text
                    "political_avg_score": avg_pol_score,
                    "stance_avg_score": avg_stance_score,
                    "url": url,
                    "original_row_id": int(idx),
                    "chunk_id": chunk_id,
                    "total_chunks": len(chunks),
                    "chunk_length": len(chunk_text),
                    "estimated_tokens": token_count
                }
            })
    
    # Batch save to ChromaDB
    ids, docs, metas = [], [], []
    
    for chunk in tqdm(all_chunks, desc="  Saving to DB", leave=False):
        ids.append(chunk["id"])
        docs.append(chunk["text"])
        metas.append(chunk["metadata"])
        
        if len(ids) >= BATCH_SIZE:
            collection.upsert(ids=ids, documents=docs, metadatas=metas)
            ids, docs, metas = [], [], []
    
    if ids:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
    
    print(f"  Completed: {len(all_chunks)} chunks saved")
    return len(all_chunks)


def main():
    print("=" * 100)
    print("Starting vector DB construction from annotated dataset")
    print("=" * 100)
    
    # Initialize ChromaDB
    print(f"\nInitializing ChromaDB: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Setup embedding function
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Collection '{COLLECTION_NAME}' ready")
    
    # Initialize tokenizer (for token-based chunking)
    print(f"\nLoading tokenizer: {EMBEDDING_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    print(f"   Max token length: {MAX_TOKENS} tokens")
    print(f"   Overlap: {OVERLAP_TOKENS} tokens")
    
    # Process CSV files
    csv_files = [f for f in os.listdir(ANNOTATED_DIR) if f.endswith('.csv')]
    print(f"\nCSV files to process: {len(csv_files)}")
    
    total_chunks = 0
    for csv_file in sorted(csv_files):
        chunks = process_csv_file(csv_file, collection, tokenizer)
        total_chunks += chunks
    
    print("\n" + "=" * 100)
    print("Vector DB construction completed!")
    print("=" * 100)
    print(f"Total chunks: {total_chunks:,}")
    print(f"Storage location: {CHROMA_DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("=" * 100)


if __name__ == "__main__":
    main()
