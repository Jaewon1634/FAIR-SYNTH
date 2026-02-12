# FAIR-SYNTH: Multi-Agent RAG Framework for Balanced Data Generation

A fairness-aware data generation framework designed to synthesize balanced datasets by mitigating political bias in large-scale web corpora through Retrieval-Augmented Generation (RAG) and Multi-Agent architecture.

## Overview

FAIR-SYNTH addresses the inherent political imbalance in large web corpora (e.g., C4). It identifies under-represented viewpoints, retrieves semantically relevant documents from over-represented perspectives, and synthesizes high-quality, balanced data using a multi-agent system. This ensures equitable representation of diverse political stances (Left/Right/Center) and viewpoints (Support/Oppose/Neutral).

## Architecture

The framework operates in a four-stage pipeline:

### 1. Topic-Aware Collection (`0_annotation/`)
- Extracts documents related to sensitive political topics (e.g., Death Penalty, Immigration).
- Filters content using keyword matching and NLI-based relevance scoring.

### 2. Multi-Persona Annotation (`0_annotation/`)
- Annotates political orientation and stance using multi-persona prompting with GPT-4.1.
- Aggregates scores from diverse personas to ensure objective labeling.
- Identifies deficits in specific viewpoint combinations (e.g., Right-Support vs. Left-Oppose).

### 3. Vector Database Construction (`1_vectordb/`)
- Chunks and embeds documents using `multilingual-e5-large-instruct`.
- Stores embeddings with rich metadata (political stance, topic) in ChromaDB.
- Enables precise retrieval of counter-viewpoint documents.

### 4. Multi-Agent Generation (`2_rag/`)
- **Retrieval**: Searches for documents from over-represented perspectives to serve as context.
- **Outline Agent**: Analyzes retrieved context and recommends a strategic outline (Content Type, Angle, Audience) for perspective switching.
- **Generation Agent**: Synthesizes high-quality, balanced text based on the outline, ensuring logical consistency and diversity.

## Key Features

- **Multi-Agent Collaboration**: Separates planning (Outline) and writing (Generation) for higher quality.
- **Perspective Switching**: Generates under-represented viewpoints by reframing over-represented contexts.
- **Symmetric Balancing**: Targets specific deficits in symmetric viewpoint pairs (e.g., balancing Left-Support with Right-Against).
- **Quality Assurance**: Evaluates generated text using Perplexity and Self-BLEU metrics.
- **Bias Mitigation Verification**: Validates fairness improvements via Political Compass and CrowS-Pairs benchmarks.

## Quick Start

1. **Install Dependencies**
   pip install -r requirements.txt
   
