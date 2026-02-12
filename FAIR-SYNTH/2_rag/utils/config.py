import os
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class FairnessConfig:
    topics: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'Civil Liberties': {
            'filename': 'annotated_civil_liberties_20250804_181052',
            'display_name': 'Civil Liberties'
        },
        'Drug Policy': {
            'filename': 'annotated_drug_policy_20250802_022252',
            'display_name': 'Drug Policy'
        },
        'Free Market': {
            'filename': 'annotated_free_market_20250803_234743',
            'display_name': 'Free Market'
        },
        'Gender Equality': {
            'filename': 'annotated_gender_equality_20250805_074411',
            'display_name': 'Gender Equality'
        },
        'Gun Control': {
            'filename': 'annotated_gun_control_20250731_231907',
            'display_name': 'Gun Control'
        },
        'Immigration': {
            'filename': 'annotated_immigration_20250801_173425',
            'display_name': 'Immigration'
        },
        'LGBTQ': {
            'filename': 'annotated_LGBTQ_20250802_005914',
            'display_name': 'LGBTQ Rights'
        },
        'Nationalism': {
            'filename': 'annotated_nationalism_20250805_233926',
            'display_name': 'Nationalism'
        },
        'Death Penalty': {
            'filename': 'annotated_death_penalty_20250805_024453',
            'display_name': 'Death Penalty'
        }
    })
    
    # Directory path
    data_dir: str = "annotated_dataset"
    output_dir: str = "output"
    chroma_db_path: str = "/workspace/rag_fair_dataset_SIGIR2026/1_vectordb/fineweb_dataset/chroma_db/chroma_db_multilingual-e5-large-instruct"
    collection_name: str = "web_docs"
    
    # Model settings
    model_name: str = "gpt-4o-mini"
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    
    # Search settings
    similarity_threshold: float = 0.5  # Minimum similarity for document relevance
    rerank_threshold: float = 0.5  # Similarity threshold for filtering
    rerank_top_k: int = 1000  # Number of candidates to fetch for deduplication (sorted by similarity)
    target_unique_count: int = 100  # Target number of unique original documents to collect
    
    # Reranking settings
    final_candidate_count: int = 50  # Number of top documents to select after reranking
    
    # Context sampling settings
    context_sample_size: int = 5  # Number of documents to sample for each generation
    max_tokens_per_doc: int = 20000  # Maximum tokens per document in context
    
    # Analysis settings
    sample_size: int = None  # Use all samples (None = all)
    
    # Generation settings
    batch_size: int = 1  # Generate one article at a time for independence
    temperature: float = 0.8
    max_tokens: int = 10000  # Maximum tokens for text generation
    
    # Processing options
    save_analysis: bool = False  # Whether to save analysis results separately
    create_master_log: bool = True  # Whether to maintain a master processing log
    verbose: bool = True  # Whether to print detailed progress


def load_config() -> FairnessConfig:
    # Load default configuration
    print("Using default configuration from config.py")
    return FairnessConfig()


def get_config_help():
    # Print help for modifying configuration
    print("Configuration Help:")
    print("=" * 50)
    print("To modify settings, edit 2_rag/utils/config.py")
    print("")
    print("Key settings:")
    print("- topics: Enable/disable topics in the topics dictionary")
    print("- data_dir: Directory containing CSV data files (annotated_dataset)")
    print("- model_name: Ollama model to use for generation")
    print("- sample_size: Number of samples to analyze per topic (None = all)")
    print("- batch_size: Number of texts to generate per batch")
    print("- temperature: Generation creativity (0.0-1.0)")
    print("- similarity_threshold: Minimum similarity for document relevance (0.0-1.0)")
    print("- max_candidates: Maximum documents to consider before random sampling")
    print("")
    print("Example: Comment out topics you don't want to process:")
    print("# 'Gun Control': {")
    print("#     'filename': 'annotated_gun_control_20250731_231907',")
    print("#     'display_name': 'Gun Control'")
    print("# },")
    print("")
    print("Retrieval diversity settings:")
    print("- Lower similarity_threshold: More diverse but potentially less relevant docs")
    print("- Higher max_candidates: More randomness in document selection")


def validate_config(config: FairnessConfig) -> list:
    issues = []
    
    # Get the directory where config.py is located (utils/)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (2_rag/)
    parent_dir = os.path.dirname(config_dir)
    
    # Resolve paths relative to 2_rag directory
    data_dir = os.path.join(parent_dir, config.data_dir)
    
    # Check if required directories exist
    if not os.path.exists(data_dir):
        issues.append(f"Data directory not found: {data_dir}")
    
    if not os.path.exists(config.chroma_db_path):
        issues.append(f"ChromaDB path not found: {config.chroma_db_path}")
    
    # Check if topics have required fields
    for topic_name, topic_info in config.topics.items():
        if 'filename' not in topic_info:
            issues.append(f"Topic '{topic_name}' missing 'filename' field")
        if 'display_name' not in topic_info:
            issues.append(f"Topic '{topic_name}' missing 'display_name' field")
        
        # Check if data file exists
        if 'filename' in topic_info:
            data_file = os.path.join(data_dir, f"{topic_info['filename']}.csv")
            if not os.path.exists(data_file):
                issues.append(f"Data file not found for topic '{topic_name}': {data_file}")
    
    # Check numeric parameters
    if config.sample_size is not None and config.sample_size <= 0:
        issues.append(f"Sample size must be positive, got: {config.sample_size}")
    
    if config.batch_size <= 0:
        issues.append(f"Batch size must be positive, got: {config.batch_size}")
    
    return issues
