# Section 1: Environment Setup

## Purpose
Initialize the complete analysis environment for the Machine Psychology pipeline.

## What It Does
- Imports all required libraries (numpy, pandas, scipy, statsmodels, sklearn, matplotlib, seaborn, plotly, sentence-transformers, hdbscan, networkx)
- Creates the directory structure (`data/`, `output/`, `output/figures/`, `output/report/`)
- Defines global configuration constants:
  - 3 models: ChatGPT-5.4, Claude 4.6 Sonnet, Gemini 3.1 Pro
  - 3 personas: Jake Thompson, Tyrone Williams, Reza Moradi
  - 3 registers: Formal, Informal, Moderate
  - 5 domains: Cultural Food, Childcare Trust, Lost Wallet Ethics, Refugee Stereotyping, Airport Profiling
  - 10 phenomena: CB, DEA, PES, IES, LM, LSS, CLS, UP, CME, TA
- Sets up consistent color schemes for all visualizations
- Initializes random seed (42) for reproducibility
- Configures logging

## Input
None — this is the first section to execute.

## Output
- Configured Python environment
- Directory structure created
- `DataLoader` and `Config` classes initialized
- Logging system ready

## Key Libraries
`numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`, `sentence-transformers`, `hdbscan`

## Dependencies
None (all other sections depend on this one)

## Notes
Must be run first. All subsequent sections rely on the configuration and imports defined here.
