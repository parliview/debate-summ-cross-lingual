# Parliview Multilingual

A comprehensive multilingual analysis project for parliamentary debates and interventions, featuring support for over 100 languages.

## Overview

This project analyzes parliamentary debates and interventions across multiple languages, providing insights into multilingual political discourse. The system processes debate data, generates summaries, and evaluates content across different languages.

## Features

- **Multilingual Support**: Analysis across 100+ languages including English, French, German, Japanese, Chinese, Spanish, and many more
- **Debate Processing**: Automated processing of parliamentary interventions and debates
- **Summary Generation**: AI-powered summarization of debate content
- **Position Reconstruction**: Analysis of political positions and stances
- **Evaluation Metrics**: Comprehensive evaluation using BERTScore and other metrics

## Project Structure

```
parliview-multilingual/
├── data/                          # Dataset and debate files
│   ├── debates/                   # Parliamentary debate data
│   │   └── CRE-*/                 # Individual debate sessions
│   │       ├── interventions/     # Debate interventions
│   │       ├── summaries/         # Generated summaries
│   │       └── reports/           # Analysis reports
│   ├── claude_data.csv           # Claude analysis data
│   └── ep_mep_activities.json    # MEP activity data
├── experiments/                   # Analysis and processing scripts
│   ├── generate_debate_summaries.py
│   ├── reconstruct_positions.py
│   ├── scrape_debates.py
│   ├── summarise_interventions.py
│   └── utils/                     # Utility functions
├── figures/                       # Generated visualizations
├── *.ipynb                        # Jupyter notebooks for analysis
└── counts.json                    # Language distribution data
```

## Language Support

The project supports analysis in over 100 languages, with the following having the highest token counts:

- **English (EN)**: 7,047,545 tokens
- **Cebuano (CEB)**: 6,116,122 tokens  
- **German (DE)**: 3,045,501 tokens
- **French (FR)**: 2,705,761 tokens
- **Swedish (SV)**: 2,615,554 tokens
- **Chinese (ZH)**: 1,497,199 tokens
- **Japanese (JA)**: 1,471,096 tokens
- **Spanish (ES)**: 2,057,696 tokens
- **Dutch (NL)**: 2,195,624 tokens
- **Russian (RU)**: 2,061,152 tokens

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/parliview-multilingual.git
cd parliview-multilingual
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the analysis notebooks:
```bash
jupyter notebook
```

## Usage

### Running Analysis

1. **Debate Summarization**:
```bash
python experiments/generate_debate_summaries.py
```

2. **Position Reconstruction**:
```bash
python experiments/reconstruct_positions.py
```

3. **Intervention Summarization**:
```bash
python experiments/summarise_interventions.py
```

### Jupyter Notebooks

- `exploration.ipynb`: Data exploration and analysis
- `eval.ipynb`: Evaluation metrics and results
- `languages.ipynb`: Language-specific analysis

## Data

The project includes parliamentary debate data from various sessions, with interventions, summaries, and reports for each debate session. The data is organized by session ID (e.g., CRE-20050221-ITEM-013).

## Evaluation

The project includes comprehensive evaluation metrics:
- BERTScore evaluation for summary quality
- Hierarchical evaluation approaches
- Zero-shot evaluation across languages
- Coefficient analysis for different models

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- European Parliament data sources
- Multilingual NLP community
- Contributors and researchers

## Contact

For questions or contributions, please open an issue or contact the maintainers.
