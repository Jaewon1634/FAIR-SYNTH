import json
import os
import sys
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZERS = {
    "Llama": "meta-llama/Llama-3.2-3B",
    "Gemma": "google/gemma-3-4b-it",
    "Qwen": "Qwen/Qwen3-4B",
}

MAX_TOKENS = 4096
OVERLAP_TOKENS = 200


def extract_prefix(text: str) -> Tuple[str, str]:
    """
    Extract prefix from text.
    Prefix is usually in the form of "This content discusses [TOPIC] from a [PERSPECTIVE] perspective [SUPPORT/AGAINST] the topic.\\n\\n"
    """
    if "\\n\\n" in text:
        parts = text.split("\\n\\n", 1)
        if len(parts) == 2:
            return parts[0] + "\\n\\n", parts[1]
    elif "\n\n" in text:
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[0] + "\n\n", parts[1]
    
    return "", text


def chunk_text_with_prefix(
    text: str,
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS
) -> List[str]:
    """
    Chunk text while preserving prefix.
    
    Args:
        text: original text
        tokenizer: tokenizer
        max_tokens: maximum token count (including prefix)
        overlap_tokens: overlap token count
    """
    prefix, content = extract_prefix(text)
    
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(full_tokens) <= max_tokens:
        return [text]
    
    if prefix:
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        prefix_token_count = len(prefix_tokens)
    else:
        prefix_token_count = 0
    
    content_tokens = tokenizer.encode(content, add_special_tokens=False)
    content_token_count = len(content_tokens)
    
    if prefix_token_count >= max_tokens:
        print(f"  Warning: prefix is too large ({prefix_token_count} tokens >= {max_tokens})")
        prefix = ""
        prefix_token_count = 0
        content_tokens = full_tokens
        content_token_count = len(content_tokens)

    available_content_tokens = max_tokens - prefix_token_count
    
    if available_content_tokens <= 0:
        return [text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < content_token_count:
        end_idx = min(start_idx + available_content_tokens, content_token_count)
        
        chunk_content_tokens = content_tokens[start_idx:end_idx]
        chunk_content = tokenizer.decode(chunk_content_tokens, skip_special_tokens=True)
        
        if prefix:
            chunk_text = prefix + chunk_content
        else:
            chunk_text = chunk_content
        
        chunks.append(chunk_text)
        
        if end_idx >= content_token_count:
            break
        
        start_idx = end_idx - overlap_tokens
        
        if start_idx < 0:
            start_idx = 0
    
    return chunks


def rechunk_jsonl(
    input_path: str,
    output_path: str,
    tokenizer_name: str,
    max_tokens: int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    hf_token: str = None
):
    """
    Rechunk JSONL file.
    
    Args:
        input_path: input JSONL file path
        output_path: output JSONL file path
        tokenizer_name: tokenizer model name
        hf_token: Hugging Face token (for gated models)
    """
    print(f"\n{'='*80}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"{'='*80}")
    
    # tokenizer loading
    print("Loading tokenizer...")
    kwargs = {"trust_remote_code": True}
    if hf_token:
        kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    print("Tokenizer loaded successfully\n")
    
    # Statistics
    total_samples = 0
    chunked_samples = 0
    total_chunks = 0
    length_stats = {
        "original": [],
        "chunked": []
    }
    
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Maximum token count: {max_tokens}, overlap: {overlap_tokens} tokens\n")
    
    # Read and process file
    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        
        lines = list(f_in)
        for line in tqdm(lines, desc="Processing"):
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if not text:
                    continue
                
                total_samples += 1
                
                # Check token count
                tokens = tokenizer.encode(text, add_special_tokens=False)
                original_token_count = len(tokens)
                length_stats["original"].append(original_token_count)
                
                # If original token count is less than max_tokens, use it directly
                if original_token_count <= max_tokens:
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total_chunks += 1
                else:
                    # Chunking needed
                    chunked_samples += 1
                    chunks = chunk_text_with_prefix(text, tokenizer, max_tokens, overlap_tokens)
                    
                    for chunk_idx, chunk_text in enumerate(chunks):
                        chunk_obj = obj.copy()
                        chunk_obj["text"] = chunk_text
                        # Add chunk index to original ID (optional)
                        if "id" in chunk_obj:
                            chunk_obj["id"] = f"{chunk_obj['id']}_chunk{chunk_idx}"
                        
                        f_out.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                        total_chunks += 1
                        
                        # Record chunk token count
                        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)
                        length_stats["chunked"].append(len(chunk_tokens))
            
            except Exception as e:
                print(f"\nWarning (line processing): {e}")
                continue
    
    # Print statistics
    print(f"\n{'='*80}")
    print("Chunking statistics")
    print(f"{'='*80}")
    print(f"Original sample count: {total_samples:,}")
    print(f"Chunked sample count: {chunked_samples:,} ({chunked_samples/total_samples*100:.1f}%)")
    print(f"Total chunk count: {total_chunks:,}")
    print(f"Average chunk count/sample: {total_chunks/total_samples:.2f}")
    
    if length_stats["original"]:
        import numpy as np
        orig_arr = np.array(length_stats["original"])
        print(f"\nOriginal token count statistics:")
        print(f"  Average: {orig_arr.mean():.1f}")
        print(f"  Median: {np.median(orig_arr):.1f}")
        print(f"  Maximum: {orig_arr.max()}")
        print(f"  {max_tokens} exceeds: {(orig_arr > max_tokens).sum()} ({(orig_arr > max_tokens).sum()/len(orig_arr)*100:.1f}%)")
    
    if length_stats["chunked"]:
        import numpy as np
        chunk_arr = np.array(length_stats["chunked"])
        print(f"\nChunk token count statistics:")
        print(f"  Average: {chunk_arr.mean():.1f}")
        print(f"  Median: {np.median(chunk_arr):.1f}")
        print(f"  Minimum: {chunk_arr.min()}")
        print(f"  Maximum: {chunk_arr.max()}")
        print(f"  {max_tokens} exceeds: {(chunk_arr > max_tokens).sum()} ({(chunk_arr > max_tokens).sum()/len(chunk_arr)*100:.1f}%)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rechunk JSONL file by token count")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--tokenizer", type=str, choices=list(TOKENIZERS.keys()), 
                       default="Llama", help="Tokenizer to use")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token (read from environment variable HF_TOKEN)")
    parser.add_argument("--max_tokens", type=int, default=MAX_TOKENS,
                       help=f"Maximum token count (default: {MAX_TOKENS})")
    parser.add_argument("--overlap_tokens", type=int, default=OVERLAP_TOKENS,
                       help=f"Overlap token count (default: {OVERLAP_TOKENS})")
    
    args = parser.parse_args()
    
    # Check
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    tokenizer_name = TOKENIZERS[args.tokenizer]
    
    rechunk_jsonl(args.input, args.output, tokenizer_name, args.max_tokens, args.overlap_tokens, hf_token)


if __name__ == "__main__":
    main()
