# Parliamentary Cross-Lingual Debate Summarisation Experiments

A research system for evaluating the quality and accuracy of structured summaries of European Parliament debates across different languages and summarisation approaches.

## Overview

This project evaluates:
- **Summary Fairness**: How well do different models generate structured summaries of parliamentary speeches? How faithfully do those summaries represent the contributions of different speakers? 
- **Language Effects**: How does speaking in different languages affect summarisation quality and accuracy?
- **Approach Comparison**: Which summarisation methods (hierarchical, incremental, zero-shot) work best?
- **Cross-Language Performance**: How well do summaries work when speakers use their native languages vs. English?

The system processes multilingual European Parliament debates where speakers may speak in their native languages (French, German, Spanish, etc.) but English translations are available, allowing for systematic comparison of summarisation quality across languages.

## Features

- **Multiple Summarisation Approaches**: 
  - Structured vs unstructured summarisation
  - Hierarchical summarisation with issue/position/argument/proposal breakdown
  - Zero-shot summarisation across languages
- **Multiple Model Support**: Ollama (local), Claude, and GPT integration for comparative evaluation
- **Domain Specific Evaluation**:
  - Local reasoning model (Qwen) extracts speaker positions from debate summaries
  - Multilingual BERTScore to compare extracted positions and original interventions


## Usage

Run experiments and evaluation scripts from the `experiments/` directory:

```sh
# 1. Generate structured summaries of individual parliamentary interventions
# Compare performance across different models and languages
python experiments/summarise_interventions.py \
    --input-dir data/debates \
    --model-type 'claude' \
    --model-name 'claude-3-sonnet-20240229' \
    --structured 

# 2. Aggregate intervention summaries into comprehensive debate reports
# Test different aggregation approaches (hierarchical, incremental, zero-shot)
python experiments/generate_debate_summaries.py \
    --input-dir data/debates \
    --model-type 'claude' \
    --model-name 'claude-3-sonnet-20240229' \
    --structured \
    --hierarchical

# 3. Reconstruct individual speaker positions from debate summaries
# Evaluate how well positions can be reconstructed from summaries
python experiments/reconstruct_positions.py \
    --model-name 'qwen3:30b-a3b' \
    --model-type 'ollama' \
    --input-dir data/debates

# 4. Evaluate summarisation quality and language effects using BERTScore
jupyter notebook eval.ipynb
```

## Project Structure

```
parliview-multilingual/
├── data/                          # Parliamentary debate dataset
│   ├── debates/                   # Individual debate sessions (CRE-*)
│   │   ├── CRE-20050221-ITEM-013/
│   │   │   ├── interventions/     # Raw intervention data (JSON)
│   │   │   ├── summaries/         # Generated intervention summaries
│   │   │   ├── reports/           # Generated debate reports
│   │   │   └── reconstructed_summaries/  # Reconstructed positions
│   │   └── ...
│   ├── claude_data.csv           # Claude analysis data
│   └── ep_mep_activities.json    # MEP activity data
├── experiments/                   # Main experiment scripts
│   ├── generate_debate_summaries.py    # Debate-level summarisation
│   ├── summarise_interventions.py     # Intervention summarisation
│   ├── reconstruct_positions.py       # Position reconstruction
│   ├── scrape_debates.py             # Data scraping utilities
│   └── utils/                         # Utility modules
│       ├── generator.py              # Debate summary generation
│       ├── summariser.py             # Intervention summarisation
│       ├── reconstructor.py          # Position reconstruction
│       ├── bertscore.py              # Evaluation metrics
│       ├── formatting.py             # Data formatting
│       └── prompts.py                # LLM prompts
├── figures/                       # Generated visualizations
├── eval.ipynb                     # Evaluation notebook
├── exploration.ipynb              # Data exploration notebook
├── languages.ipynb                # Language analysis notebook
└── counts.json                    # Language distribution data
```

## Data Structure

The `data/debates/` directory contains European Parliament debate sessions with the following structure:

```
data/debates/CRE-20060118-ITEM-008/
├── interventions/                 # Raw parliamentary speech data
│   ├── 3-230.json               # Individual speech files with speaker, text, language
│   ├── 3-231.json               # Contains: speaker, english, original, lang, agenda_item
│   └── ...
├── summaries/                    # Generated structured summaries of interventions
│   ├── 3-230-claude-sonnet-4-20250514-structured.json
│   ├── 3-231-claude-sonnet-4-20250514-structured.json
│   └── ...                      # Contains: headline, issueSum, positionSum, argSum, propSum, quotes
├── reports/                      # Generated comprehensive debate reports
│   ├── CRE-20060118-ITEM-008_claude_structured_en.json
│   ├── CRE-20060118-ITEM-008_claude_structured_org.json
│   └── ...                      # Contains: contributions, summary, metadata
└── reconstructed_summaries/      # Reconstructed individual speaker positions
    └── qwen3:30b-a3b/
        ├── CRE-20060118-ITEM-008_claude_structured_en.json
        └── CRE-20060118-ITEM-008_claude_structured_org.json
```

### Data Format

**Intervention files** contain:
- `speaker`: Name and party of the speaker
- `english`: English translation of the speech
- `original`: Original language version of the speech
- `lang`: Language code (EN, FR, DE, etc.)
- `agenda_item`: Topic of the debate
- `debate_id`: Unique identifier for the debate session
- `intervention_id`: Unique identifier for the speech

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/parliview-multilingual.git
cd parliview-multilingual
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# For Ollama (local models)
export OLLAMA_API_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="mistral"

# For Claude API
export ANTHROPIC_API_KEY="your-api-key"
export CLAUDE_MODEL="claude-3-sonnet-20240229"

# For OpenAI API
export OPENAI_API_KEY="your-api-key"
export GPT_MODEL="gpt-4-turbo-preview"
```
