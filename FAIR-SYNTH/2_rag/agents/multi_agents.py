import os
import pandas as pd
import chromadb
import random
import json
import re
from typing import List, Dict, TypedDict, Annotated, Sequence
from collections import defaultdict, Counter
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tracers import LangChainTracer
from langsmith import Client
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# LangSmith settings (disabled to reduce log size and network errors)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGCHAIN_API_KEY", None)

# Configuration
ANNOTATED_DATASET_DIR = "/workspace/rag_fair_dataset_SIGIR2026/2_rag/annotated_dataset"
CHROMA_DB_PATH = "/workspace/rag_fair_dataset_SIGIR2026/1_vectordb/C4_dataset/chroma_db"
COLLECTION_NAME = "c4_docs"
OUTPUT_DIR = "/workspace/rag_fair_dataset_SIGIR2026/2_rag/output"
AGENTS_OUTPUT_DIR = "/workspace/rag_fair_dataset_SIGIR2026/2_rag/agents_output"
OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_BASE_URL = "http://localhost:11434"

# query template
QUERY_TEMPLATE = "Article about {topic} from {political} {stance} perspective"

# Fixed k value
FIXED_K = 5

# Content types for diversity
CONTENT_TYPES = [
    "blog",              # Blog post - personal perspective
    "news",              # News article - event reporting with angle
    "column",            # Opinion column - regular commentary
    "editorial",         # Editorial - institutional position
    "report",            # Analysis report - in-depth investigation
    "opinion",           # Opinion piece - argument-driven
    "analysis",          # Policy analysis - detailed examination
    "feature",           # Feature article - narrative with depth
    "commentary",        # Commentary - expert interpretation
    "essay",             # Essay - thoughtful exploration
    "review",            # Review/critique - evaluation
    "social_media",      # Social media post - brief but impactful
    "open_letter",       # Open letter - direct appeal
    "speech",            # Speech/address - persuasive oration
    "policy_brief"       # Policy brief - actionable recommendations
]

STYLE_GUIDES = [
    "academic and analytical tone",           # Evidence-based, scholarly
    "passionate and persuasive tone",         # Emotional appeal, conviction
    "objective and fact-based tone",          # Neutral presentation, data-driven
    "critical and confrontational tone",      # Sharp critique, challenging
    "conversational and accessible tone",     # Easy to understand, relatable
    "urgent and activist tone"                # Call to action, mobilizing
]

TARGET_AUDIENCES = [
    "general public",                         # Broad audience
    "policy makers and legislators",          # Decision makers
    "activists and advocates",                # Movement participants
    "academic and research community",        # Scholars and experts
    "young adults and students",              # Youth demographic
    "concerned citizens",                     # Engaged public
    "business and corporate leaders",         # Economic stakeholders
    "media and journalists"                   # Opinion shapers
]

ANGLES = [
    "economic impact focused",                # Financial/market effects
    "social justice focused",                 # Equity and fairness
    "individual rights focused",              # Personal freedom/autonomy
    "public safety focused",                  # Security and protection
    "moral and ethical focused",              # Values and principles
    "practical policy focused",               # Implementation and feasibility
    "historical context focused",             # Past precedents and lessons
    "future implications focused",            # Long-term consequences
    "comparative perspective focused",        # Cross-country/cross-context
    "grassroots movement focused",            # Bottom-up organizing
    "institutional reform focused",           # Systemic change
    "cultural and identity focused"           # Community and belonging
]

# Symmetric pairs for deficit calculation
SYMMETRIC_PAIRS = [
    ('Left-Against', 'Right-Support'),
    ('Left-Support', 'Right-Against'),
    ('Left-Neutral', 'Right-Neutral'),
    ('Center-Against', 'Center-Support'),
]


def _get_main_perspective_from_row(row: pd.Series, ann_cols: List[str]) -> tuple[str, str] | None:
    """
    Determine (political, stance) for a single annotated CSV row.

    Preference order:
    1) political_major / stance_major columns if present and non-empty
    2) Majority vote over annotation JSON columns (gpt-4.1_*, claude-sonnet-*)
    """
    pol = row.get("political_major")
    st = row.get("stance_major")
    if isinstance(pol, str) and pol.strip() and isinstance(st, str) and st.strip():
        return pol.strip(), st.strip()

    labels: List[tuple[str, str]] = []
    for col in ann_cols:
        val = row.get(col)
        if not isinstance(val, str) or not val.strip():
            continue
        try:
            ann = json.loads(val)
        except Exception:
            continue
        pol2 = ann.get("Political", {}).get("label")
        st2 = ann.get("Stance", {}).get("label")
        if pol2 and st2:
            labels.append((str(pol2).strip(), str(st2).strip()))

    if not labels:
        return None
    cnt = Counter(labels)
    (major_pol, major_st), _ = cnt.most_common(1)[0]
    return major_pol, major_st


# ==================== State Definition ====================
def extract_base_topic(topic_with_suffix: str) -> str:
    parts = topic_with_suffix.split("_")
    # Strip timestamp suffix like YYYYMMDD_HHMMSS
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        base = "_".join(parts[:-2])
    else:
        base = topic_with_suffix
    
    # Map to ChromaDB topic names
    mapping = {
        "LGBTQ": "LGBTQ",
        "civil_liberties": "civil liberties",
        "death_penalty": "death penalty",
        "drug_policy": "drug policy",
        "free_market": "free-market",
        "gender_equality": "gender equality",
        "gun_control": "gun control",
        "immigration": "immigration",
        "nationalism": "nationalism",
    }
    if base in mapping:
        return mapping[base]
    # Fallback: replace underscores with spaces
    return base.replace("_", " ")


class PipelineState(TypedDict, total=False):
    """Complete pipeline state"""
    # Dataset loading
    dataset: Dict[str, pd.DataFrame]  # topic (full) -> DataFrame
    topics: List[str]                 # list of full topic names (with timestamp)
    topic_base_map: Dict[str, str]    # full topic -> base topic used in Chroma
    
    # Distribution analysis
    topic_distributions: Dict[str, Dict[str, int]]  # topic -> {combo: count}
    deficits: List[Dict]  # [{topic, political, stance, deficit, ...}]
    current_deficit_idx: int
    
    # Current generation context
    current_topic: str
    current_political: str
    current_stance: str
    current_deficit: int
    current_iteration: int  # Within current deficit
    
    # Document search
    searched_documents: List[Dict]  # [{original_text, metadata, distance}]
    sampled_originals: List[str]  # Original texts (not chunks)
    context_political: str        # political label of context docs (may be symmetric)
    context_stance: str           # stance label of context docs (may be symmetric)
    
    # Agent outputs
    outline: Dict
    generated_text: str
    
    # Results
    generation_results: List[Dict]  # All generated items
    
    # Statistics
    total_generated: int
    content_type_counter: Dict[str, int]
    style_counter: Dict[str, int]
    angle_counter: Dict[str, int]


# ==================== Node Functions ====================

class RAGPipeline:
    """Complete RAG pipeline using LangGraph - Pure LLM approach"""
    
    def __init__(self):
        # ChromaDB connection
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        
        # LLM initialization
        self.llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.8
        )
        
        # Build graph
        self.graph = self._build_graph()
        
        print(f"RAGPipeline initialized (Pure LLM approach)")
        print(f"   - Dataset: {ANNOTATED_DATASET_DIR}")
        print(f"   - ChromaDB: {CHROMA_DB_PATH}/{COLLECTION_NAME}")
        print(f"   - LLM: {OLLAMA_MODEL}")
        print(f"   - Strategy: Pure LLM (No few-shot, No rule-based reasoning)")
        print(f"   - Fixed Query: {QUERY_TEMPLATE}")
        print(f"   - Fixed k: {FIXED_K}")
    
    def _build_graph(self) -> StateGraph:
        """Build the complete LangGraph workflow"""
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("load_dataset", self.load_dataset_node)
        workflow.add_node("analyze_distribution", self.analyze_distribution_node)
        workflow.add_node("search_documents", self.search_documents_node)
        workflow.add_node("outline_generation", self.outline_generation_node)
        workflow.add_node("content_generation", self.content_generation_node)
        workflow.add_node("save_result", self.save_result_node)
        
        # Define edges
        workflow.add_edge(START, "load_dataset")
        workflow.add_edge("load_dataset", "analyze_distribution")
        workflow.add_edge("analyze_distribution", "search_documents")
        workflow.add_edge("search_documents", "outline_generation")
        workflow.add_edge("outline_generation", "content_generation")
        workflow.add_edge("content_generation", "save_result")
        
        # Conditional edge from save_result
        def should_continue(state: PipelineState) -> str:
            """Check if more generations are needed"""
            current_deficit_idx = state.get("current_deficit_idx", 0)
            current_iteration = state.get("current_iteration", 0)
            deficits = state.get("deficits", [])
            
            if current_deficit_idx >= len(deficits):
                return "end"
            
            current_deficit_item = deficits[current_deficit_idx]
            needed = current_deficit_item['deficit']
            
            if current_iteration < needed:
                return "search_documents"
            else:
                # Move to next deficit
                if current_deficit_idx + 1 < len(deficits):
                    return "search_documents"
                else:
                    return "end"
        
        workflow.add_conditional_edges(
            "save_result",
            should_continue,
            {
                "search_documents": "search_documents",
                "end": END
            }
        )
        
        # Compile workflow without checkpointing to avoid serializing large DataFrames
        return workflow.compile()
    
    # ==================== Node 1: Load Dataset ====================
    def load_dataset_node(self, state: PipelineState) -> PipelineState:
        """Load annotated dataset files"""
        print("\n" + "="*60)
        print("[Node 1: Load Dataset]")
        print("="*60)
        
        dataset: Dict[str, pd.DataFrame] = {}
        topics: List[str] = []
        topic_base_map: Dict[str, str] = {}
        
        # Load all CSV files
        for filename in os.listdir(ANNOTATED_DATASET_DIR):
            if filename.endswith('.csv'):
                topic_full = filename.replace('annotated_', '').replace('.csv', '')
                base_topic = extract_base_topic(topic_full)
                filepath = os.path.join(ANNOTATED_DATASET_DIR, filename)
                df = pd.read_csv(filepath)
                dataset[topic_full] = df
                topics.append(topic_full)
                topic_base_map[topic_full] = base_topic
                print(f"  Loaded {topic_full} (base topic: {base_topic}): {len(df)} rows")
        
        print(f"\nTotal topics: {len(topics)}")
        
        return {
            "dataset": dataset,
            "topics": topics,
            "topic_base_map": topic_base_map,
            "generation_results": [],
            "total_generated": 0,
            "content_type_counter": defaultdict(int),
            "style_counter": defaultdict(int),
            "angle_counter": defaultdict(int),
            "current_deficit_idx": 0,
            "current_iteration": 0
        }
    
    # ==================== Node 2: Analyze Distribution ====================
    def analyze_distribution_node(self, state: PipelineState) -> PipelineState:
        """Analyze distribution and calculate deficits (based on original annotated CSVs)"""
        print("\n" + "="*60)
        print("[Node 2: Analyze Distribution & Calculate Deficits]")
        print("="*60)
        
        topics = state["topics"]  # full topic names
        topic_distributions = {}
        all_deficits = []
        
        for topic_full in topics:
            print(f"\n[Topic: {topic_full} | distribution from original annotated CSV]")
            df = state["dataset"][topic_full]

            # Find annotation columns (for majority voting fallback)
            ann_cols = [c for c in df.columns if "gpt-4.1_" in c or "claude-sonnet" in c]

            # Count by combination from original rows only
            combination_counts = Counter()
            for _, row in df.iterrows():
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                persp = _get_main_perspective_from_row(row, ann_cols)
                if not persp:
                    continue
                political, stance = persp
                combo = f"{political}-{stance}"
                combination_counts[combo] += 1
            
            # Store distribution keyed by full topic name
            topic_distributions[topic_full] = dict(combination_counts)
            
            # Print distribution
            print(f"  Distribution:")
            for combo, count in sorted(combination_counts.items()):
                print(f"    {combo}: {count}")
            
            # Calculate deficits for symmetric pairs
            for combo1, combo2 in SYMMETRIC_PAIRS:
                count1 = combination_counts.get(combo1, 0)
                count2 = combination_counts.get(combo2, 0)
                
                if count1 > count2:
                    underrep = combo2
                    deficit = count1 - count2
                elif count2 > count1:
                    underrep = combo1
                    deficit = count2 - count1
                else:
                    continue
                
                political, stance = underrep.split('-')
                
                all_deficits.append({
                    'topic': topic_full,
                    'political': political,
                    'stance': stance,
                    'combination': underrep,
                    'current_count': min(count1, count2),
                    'target_count': max(count1, count2),
                    'deficit': deficit,
                    'pair': (combo1, combo2)
                })
                
                print(f"  Deficit found: {underrep} needs +{deficit} (current: {min(count1, count2)}, target: {max(count1, count2)})")
        
        print(f"\n{'='*60}")
        print(f"Total deficits to generate: {len(all_deficits)}")
        print(f"Total items to generate: {sum(d['deficit'] for d in all_deficits)}")
        print(f"{'='*60}")
        
        return {
            "topic_distributions": topic_distributions,
            "deficits": all_deficits
        }
    
    # ==================== Node 3: Search Documents ====================
    def search_documents_node(self, state: PipelineState) -> PipelineState:
        """Search vector DB and sample original documents"""
        deficits = state["deficits"]
        current_deficit_idx = state.get("current_deficit_idx", 0)
        current_iteration = state.get("current_iteration", 0)
        
        # Check if we need to move to next deficit
        if current_iteration > 0:
            current_deficit_item = deficits[current_deficit_idx]
            if current_iteration >= current_deficit_item['deficit']:
                current_deficit_idx += 1
                current_iteration = 0
        
        if current_deficit_idx >= len(deficits):
            return {"current_deficit_idx": current_deficit_idx}
        
        deficit_item = deficits[current_deficit_idx]
        topic_full = deficit_item['topic']
        topic_base_map = state.get("topic_base_map", {})
        base_topic = topic_base_map.get(topic_full, topic_full)
        political = deficit_item['political']
        stance = deficit_item['stance']
        
        print("\n" + "="*60)
        print(f"[Node 3: Search Documents]")
        print(f"Deficit {current_deficit_idx + 1}/{len(deficits)}: {topic_full} (base: {base_topic}) - {political}-{stance}")
        print(f"Iteration {current_iteration + 1}/{deficit_item['deficit']}")
        print("="*60)
        
        # Determine symmetric (over-represented) combo to use as context
        pair = deficit_item.get("pair")
        target_combo = f"{political}-{stance}"
        context_political = political
        context_stance = stance
        
        if pair:
            combo1, combo2 = pair
            symmetric_combo = combo2 if combo1 == target_combo else combo1
            sym_pol, sym_stance = symmetric_combo.split("-")
            context_political, context_stance = sym_pol, sym_stance
            print(f"  Using symmetric combo {symmetric_combo} as context for target {target_combo}")
        else:
            print("  WARNING: No symmetric pair information available. Using target combo as context.")
        
        # Fixed query (no randomization) - still phrased in terms of target perspective
        query = QUERY_TEMPLATE.format(topic=base_topic, political=political, stance=stance)
        print(f"  Query (LLM target perspective): {query}")
        
        # Always search with context (usually symmetric) combo
        where_filter = {
            "$and": [
                {"topic": {"$eq": base_topic}},
                {"political_major": {"$eq": context_political}},
                {"stance_major": {"$eq": context_stance}}
            ]
        }
        
        all_filtered = self.collection.get(
            where=where_filter,
            include=['metadatas']
        )
        total_filtered = len(all_filtered['metadatas']) if all_filtered['metadatas'] else 0
        print(f"  Total documents matching context combo {context_political}-{context_stance}: {total_filtered}")
        
        if total_filtered == 0:
            print("  WARNING: No documents found for context combo. Skipping generation for this iteration.")
            return {
                "current_topic": topic_full,
                "current_political": political,
                "current_stance": stance,
                "current_query": query,
                "current_deficit": deficit_item['deficit'],
                "current_deficit_idx": current_deficit_idx,
                "current_iteration": current_iteration,
                "searched_documents": [],
                "sampled_originals": []
            }
        
        # Query with top 50%
        n_results = min(total_filtered, 10000)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results.get('distances') else []
        
        # Select top 50%
        num_to_select = max(int(len(documents) * 0.5), 5)
        num_to_select = min(num_to_select, len(documents))
        
        doc_list = [
            {
                'chunk': doc,
                'original_text': meta.get('original_text', ''),
                'metadata': meta,
                'distance': dist
            }
            for doc, meta, dist in zip(documents[:num_to_select], 
                                      metadatas[:num_to_select],
                                      distances[:num_to_select] if distances else [0]*num_to_select)
        ]
        
        print(f"  Selected top 50%: {num_to_select} documents")
        
        # Group by original_text
        text_groups = defaultdict(list)
        for doc in doc_list:
            original_text = doc['original_text']
            if original_text:
                text_groups[original_text].append(doc)
        
        unique_originals = list(text_groups.keys())
        print(f"  Unique original documents: {len(unique_originals)}")
        
        # Sample FIXED_K original texts (no randomization of k)
        k = FIXED_K
        
        if len(unique_originals) < k:
            print(f"  WARNING: Only {len(unique_originals)} unique documents (requested: {k})")
            sampled_originals = unique_originals
        else:
            sampled_originals = random.sample(unique_originals, k)
        
        print(f"  Sampled {len(sampled_originals)} original documents (k={k})")
        
        return {
            "current_topic": topic_full,
            "current_political": political,
            "current_stance": stance,
            "current_query": query,
            "context_political": context_political,
            "context_stance": context_stance,
            "current_deficit": deficit_item['deficit'],
            "current_deficit_idx": current_deficit_idx,
            "current_iteration": current_iteration,
            "searched_documents": doc_list,
            "sampled_originals": sampled_originals
        }
    
    # ==================== Node 4: Outline Generation ====================
    def outline_generation_node(self, state: PipelineState) -> PipelineState:
        """Generate content outline (Agent 1) - Pure LLM approach"""
        print("\n" + "="*60)
        print("[Node 4: Outline Generation - Agent 1]")
        print("="*60)

        # If there is no current topic (no deficits), skip generation
        if "current_topic" not in state:
            print("  No current topic in state (probably no deficits). Skipping outline generation.")
            return {"outline": {}}
        
        topic = state["current_topic"]
        political = state["current_political"]
        stance = state["current_stance"]
        context_political = state.get("context_political", political)
        context_stance = state.get("context_stance", stance)
        sampled_originals = state["sampled_originals"]
        iteration = state["current_iteration"]
        
        if not sampled_originals:
            print("  ERROR: No documents to work with")
            return {"outline": {}}
        
        # Use original texts (not chunks)
        context_docs = sampled_originals
        
        # Format context (variable length for diversity)
        doc_length = random.randint(1000, 2000)
        context_text = "\n\n".join([
            f"[Reference Document {i+1}]\n{doc[:doc_length]}..."
            for i, doc in enumerate(context_docs)
        ])
        
        # Provide options (not forcing)
        available_content_types = ", ".join(CONTENT_TYPES)
        available_angles = ", ".join(ANGLES)
        available_audiences = ", ".join(TARGET_AUDIENCES)
        
        print(f"  Available content types: {len(CONTENT_TYPES)}")
        print(f"  Available angles: {len(ANGLES)}")
        print(f"  Available audiences: {len(TARGET_AUDIENCES)}")
        
        # System prompt - context may be from opposite perspective
        system_prompt = f"""You are an experienced content strategist and analyst.

        Your task is to:
        1. Carefully read and analyze the provided reference documents
        2. Understand what arguments, perspectives, and information they present about {topic}, mostly from a {context_political} {context_stance} perspective
        3. Determine the most appropriate content format, angle, and audience based on document characteristics
        4. Plan a creative content piece that clearly expresses {political} {stance} position on {topic}, responding to or reframing the reference documents

        Important guidelines:
        - Treat the reference documents as source material, not as the final stance you must defend
        - If documents support {context_political} {context_stance}, you should critically analyze and respond from {political} {stance} perspective
        - Let the documents guide your choice of format, angle, and audience (data-heavy, personal stories, news-oriented, etc.)
        - If documents are data-heavy: consider analysis/report formats with economic or policy angles
        - If documents are personal stories: consider essay/feature formats with individual rights or cultural angles
        - If documents are news-oriented: consider news/column formats with timely angles
        - Choose the content type, angle, and audience that naturally fit the document content and your {political} {stance} response
        - Be creative and diverse in your approach
        """
                
        # User prompt
        user_prompt = f"""Here are reference documents about {topic} (mostly from a {context_political} {context_stance} perspective):

        {context_text}

        Task:
        Analyze these documents and plan content that expresses {political} {stance} position on {topic}, either by:
        - directly defending {political} {stance} using evidence and narratives from the documents, or
        - critically responding to and reframing the {context_political} {context_stance} arguments in the documents.

        Step 1 - Analyze the documents:
        - What is the primary focus of these documents? (data/policy, personal stories, news events, cultural themes, etc.)
        - What evidence or arguments do they present?
        - What emotional tone or style do they have?
        - How do their arguments align with {context_political} {context_stance} perspective?

        Step 2 - Choose appropriate format and approach:
        Based on your analysis, select the most fitting options:

        Available content types: {available_content_types}
        Available approach angles: {available_angles}
        Available target audiences: {available_audiences}

        Step 3 - Plan the content:
        - Create a title that captures both the content and your chosen angle
        - Identify key points from the documents that you will use, support, or challenge
        - Explain your reasoning for the choices you made, including how you are responding to {context_political} {context_stance} perspective from a {political} {stance} position

        Respond in JSON format:
        {{
        "content_type": "your selected content type from the available options",
        "title": "specific and creative title",
        "angle": "your selected angle from the available options",
        "target_audience": "your selected audience from the available options",
        "key_points": ["key point 1 based on documents", "key point 2 based on documents", "key point 3 based on documents"],
        "reasoning": "explain why you chose this specific content type, angle, and audience based on what you saw in the documents (2-3 sentences)"
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Parse JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            outline = json.loads(content)
            
            print(f"  Generated outline:")
            print(f"    - Type: {outline.get('content_type')}")
            print(f"    - Title: {outline.get('title', '')[:60]}...")
            if 'reasoning' in outline:
                print(f"    - Reasoning: {outline.get('reasoning', '')[:100]}...")
            
            return {"outline": outline}
            
        except Exception as e:
            print(f"  ERROR: Failed to generate outline: {e}")
            return {"outline": {}}
    
    # ==================== Node 5: Content Generation ====================
    def content_generation_node(self, state: PipelineState) -> PipelineState:
        """Generate actual content (Agent 2) - Pure LLM approach"""
        print("\n" + "="*60)
        print("[Node 5: Content Generation - Agent 2]")
        print("="*60)

        # If there is no current topic (no deficits), skip generation
        if "current_topic" not in state:
            print("  No current topic in state (probably no deficits). Skipping content generation.")
            return {"generated_text": ""}
        
        topic = state["current_topic"]
        political = state["current_political"]
        stance = state["current_stance"]
        sampled_originals = state["sampled_originals"]
        outline = state.get("outline", {})
        iteration = state["current_iteration"]
        
        if not outline or not sampled_originals:
            print("  ERROR: Missing outline or documents")
            return {"generated_text": ""}
        
        # Use original texts
        context_docs = sampled_originals
        
        # Style forcing (for diversity)
        style_counter = state.get("style_counter", defaultdict(int))
        least_used_styles = sorted(STYLE_GUIDES, key=lambda x: style_counter[x])
        forced_style = least_used_styles[iteration % len(STYLE_GUIDES)]
        
        # Temperature variation
        varied_temp = random.uniform(0.7, 0.9)
        varied_llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=varied_temp
        )
        
        # Format context (different length from Agent 1 for diversity)
        doc_length = random.randint(1500, 2500)
        context_text = "\n\n".join([
            f"[Reference Document {i+1}]\n{doc[:doc_length]}..."
            for i, doc in enumerate(context_docs)
        ])
        
        # Parse outline
        content_type = outline.get('content_type', 'article')
        title = outline.get('title', '')
        angle = outline.get('angle', '')
        target_audience = outline.get('target_audience', 'general public')
        key_points = outline.get('key_points', [])
        reasoning = outline.get('reasoning', '')
        
        # Length guide based on context
        avg_length = sum(len(doc) for doc in context_docs) / len(context_docs)
        if avg_length > 2000:
            min_length, max_length = random.randint(3000, 4000), random.randint(7000, 9000)
        elif avg_length < 1000:
            min_length, max_length = random.randint(1500, 2500), random.randint(5000, 7000)
        else:
            min_length, max_length = random.randint(2000, 3000), random.randint(6000, 8000)
        
        print(f"  Style: {forced_style}")
        print(f"  Temperature: {varied_temp:.2f}")
        print(f"  Target length: {min_length}-{max_length}")
        
        # System prompt
        writer_roles = [
            f"{content_type} professional writer",
            f"{content_type} writer and journalist",
            f"experienced {content_type} writer",
            f"creative {content_type} writer",
            f"{content_type} expert and writer"
        ]
        dynamic_writer_role = writer_roles[iteration % len(writer_roles)]
        
        system_prompt = (
            f"You are a {dynamic_writer_role}.\n\n"
            f"Your task is to write content that clearly expresses {stance} position on {topic} "
            f"from {political} political perspective.\n\n"
            "Important guidelines:\n"
            "- Reference documents are for context and inspiration, do not copy directly\n"
            "- The reference documents may mostly reflect an opposite or different perspective; "
            f"you must respond from {political} {stance} viewpoint\n"
            f"- You can use data, stories, and arguments from the reference documents, but reinterpret or "
            f"challenge them as needed to support {political} {stance}\n"
            "- Write original and natural content\n"
            f"- {political} perspective and {stance} position must be clear throughout\n"
            f"- Tone: {forced_style}\n"
            f"- Target audience: {target_audience}\n"
            f"- Length: {min_length}-{max_length} characters\n"
            f"- Format must match {content_type} conventions\n"
        )
        
        # User prompt 
        key_points_text = "\n".join([f"  * {kp}" for kp in key_points])
        
        user_prompt = f"""Content planning information:
        - Format: {content_type}
        - Title: {title}
        - Approach angle: {angle}
        - Target audience: {target_audience}
        - Tone: {forced_style}
        - Key points to cover:
        {key_points_text}

        Planning reasoning: {reasoning}

        Reference documents for context:
        {context_text}

        Now write a complete {content_type} that:
        1. Expresses {political} {stance} position on {topic}
        2. Uses the {angle} approach
        3. Incorporates the key points above (supporting, reframing, or challenging the reference documents as needed)
        4. Follows {content_type} format conventions
        5. Maintains {forced_style}
        6. Targets {target_audience}

        Write the content directly. Do not include meta-commentary or explanations about what you're doing.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = varied_llm.invoke(messages)
            generated_text = response.content.strip()
            
            # Clean up markdown artifacts
            generated_text = re.sub(r'^```[\w]*\n', '', generated_text)
            generated_text = re.sub(r'\n```$', '', generated_text)
            
            print(f"  Generated content length: {len(generated_text)} characters")
            
            # Update style counter
            style_counter[forced_style] += 1
            
            return {
                "generated_text": generated_text,
                "style_counter": style_counter
            }
            
        except Exception as e:
            print(f"  ERROR: Failed to generate content: {e}")
            return {"generated_text": ""}
    
    # ==================== Node 6: Save Result ====================
    def save_result_node(self, state: PipelineState) -> PipelineState:
        """Save generation result and update counters"""
        print("\n" + "="*60)
        print("[Node 6: Save Result]")
        print("="*60)
        
        generated_text = state.get("generated_text", "")
        outline = state.get("outline", {})
        
        if not generated_text or not outline:
            print("  Skipping save (no content generated)")
            # Still increment iteration
            return {
                "current_iteration": state["current_iteration"] + 1
            }
        
        # Prepare result (include query and contexts for Ragas faithfulness evaluation)
        result = {
            'topic': state["current_topic"],
            'political': state["current_political"],
            'stance': state["current_stance"],
            'text': generated_text,
            'query': state.get("current_query", ""),
            'retrieved_contexts': list(state.get("sampled_originals", [])),
            'content_type': outline.get('content_type', ''),
            'title': outline.get('title', ''),
            'angle': outline.get('angle', ''),
            'target_audience': outline.get('target_audience', ''),
            'key_points': outline.get('key_points', []),
            'reasoning': outline.get('reasoning', ''),
            'num_context_docs': len(state.get("sampled_originals", []))
        }
        
        # Update results
        generation_results = state.get("generation_results", [])
        generation_results.append(result)
        
        # Update counters
        content_type_counter = state.get("content_type_counter", defaultdict(int))
        angle_counter = state.get("angle_counter", defaultdict(int))
        
        content_type_counter[outline.get('content_type', '')] += 1
        angle_counter[outline.get('angle', '')] += 1
        
        total_generated = state.get("total_generated", 0) + 1
        
        print(f"  Saved result #{total_generated}")
        print(f"  Topic: {result['topic']}")
        print(f"  Combination: {result['political']}-{result['stance']}")
        print(f"  Type: {result['content_type']}")
        
        return {
            "generation_results": generation_results,
            "total_generated": total_generated,
            "content_type_counter": content_type_counter,
            "angle_counter": angle_counter,
            "current_iteration": state["current_iteration"] + 1
        }
    
    # ==================== Pipeline Execution ====================
    
    def run(self):
        """Execute the complete pipeline"""
        print("\n" + "="*80)
        print("STARTING COMPLETE RAG PIPELINE (Pure LLM Approach)")
        print("="*80)
        
        # Initialize state
        initial_state = PipelineState()
        
        # Run graph
        # Increase recursion_limit sufficiently to cover many topics * deficits * iterations
        # Set very high to practically avoid GraphRecursionError for this pipeline.
        final_state = self.graph.invoke(
            initial_state,
            config={
                "recursion_limit": 100000,
                "configurable": {"thread_id": "1"},
            },
        )
        
        # Print final statistics
        print("\n" + "="*80)
        print("PIPELINE COMPLETED")
        print("="*80)
        print(f"Total generated: {final_state.get('total_generated', 0)}")
        print(f"\nContent type distribution:")
        for ctype, count in sorted(final_state.get('content_type_counter', {}).items()):
            print(f"  {ctype}: {count}")
        print(f"\nAngle distribution:")
        for angle, count in sorted(final_state.get('angle_counter', {}).items()):
            print(f"  {angle}: {count}")
        
        # Save results
        results = final_state.get('generation_results', [])
        if results:
            # Save JSON results by topic
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Group by topic
            by_topic = defaultdict(list)
            for result in results:
                by_topic[result['topic']].append(result)
            
            # Save each topic as JSON and merged CSV
            os.makedirs(AGENTS_OUTPUT_DIR, exist_ok=True)
            
            for topic, topic_results in by_topic.items():
                # 1) Save generated results as JSON
                output_file = os.path.join(OUTPUT_DIR, f"generated_{topic}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(topic_results, f, ensure_ascii=False, indent=2)
                print(f"\nSaved {len(topic_results)} generated results to: {output_file}")
                
                # 2) Load original annotated dataset for this topic
                original_path = os.path.join(ANNOTATED_DATASET_DIR, f"annotated_{topic}.csv")
                if os.path.exists(original_path):
                    try:
                        original_df = pd.read_csv(original_path)
                        print(f"Loaded original dataset for topic '{topic}': {len(original_df)} rows")
                    except Exception as e:
                        print(f"WARNING: Failed to load original dataset for topic '{topic}': {e}")
                        original_df = pd.DataFrame()
                else:
                    print(f"WARNING: Original dataset file not found for topic '{topic}': {original_path}")
                    original_df = pd.DataFrame()
                
                # 3) Convert generated results to DataFrame
                generated_df = pd.DataFrame(topic_results)
                
                # 4) Merge original and generated data
                if not original_df.empty:
                    merged_df = pd.concat([original_df, generated_df], ignore_index=True, sort=False)
                else:
                    merged_df = generated_df
                
                # 5) Save merged CSV to agents_output with topic in filename
                merged_path = os.path.join(AGENTS_OUTPUT_DIR, f"{topic}_with_generated.csv")
                try:
                    merged_df.to_csv(merged_path, index=False)
                    print(f"Saved merged original + generated data for topic '{topic}' to: {merged_path}")
                except Exception as e:
                    print(f"WARNING: Failed to save merged CSV for topic '{topic}': {e}")
        
        return final_state


if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.run()
