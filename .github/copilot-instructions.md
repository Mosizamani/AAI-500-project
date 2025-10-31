# Himalayan Expedition Success Predictor - AI Coding Agent Instructions

## Project Overview
This is a data science project analyzing historical Himalayan expedition data to predict summit success using logistic regression and GLM models. The core analysis focuses on expeditions above 7,000m and identifies key factors like oxygen use, route type, and seasonal patterns.

## Key Architecture & Data Flow

### Single Notebook Analysis Pipeline
- **Main workflow**: `EDA_for_Himalayan_Summit_Success_glm.ipynb` contains the complete analysis pipeline
- **Data sources**: Four CSV files in `data/` directory:
  - `expeditions.csv` - Primary dataset with expedition records (10K+ rows)
  - `deaths.csv`, `peaks.csv`, `summiters.csv` - Supporting datasets
- **Output artifacts**: Visualizations saved to `overleaf/` for LaTeX publication

### Data Processing Pattern
```python
# Standard workflow pattern used throughout:
1. Load raw expeditions.csv
2. Drop "bad_features" list (expedition metadata not predictive)
3. Filter for max_elev_reached > 7000m
4. Transform mbrs_summited to binary success variable
5. One-hot encode categorical variables (especially seasons)
6. Split 80/20 train/test with random_state=42
```

## Critical Project Conventions

### Feature Engineering Approach
- **Target variable**: `success` (binary) derived from `mbrs_summited > 0`
- **Key predictors**: Focus on `is_o2_climbing`, `is_standard_rte`, `season_*` dummy variables
- **Feature filtering**: Extensive `bad_features` list removes non-predictive expedition metadata
- **Altitude threshold**: Analysis restricted to expeditions above 7,000m for high-stakes focus

### Model Evaluation Standards
- **Primary metrics**: Accuracy (~78%), AUC (~0.79), plus Peirce Skill Score (PSS) and Heidke Skill Score (HSS)
- **Visualization outputs**: All plots saved to `overleaf/` directory as high-DPI PNGs for publication
- **Cross-validation**: Not implemented (identified as future improvement in README)

### Code Organization Pattern
```python
# Consistent function structure:
def plot_success_rate(df, column, title):
    """Helper functions for visualization with standardized parameters"""
    # Calculate success rates
    # Create publication-ready plots
    # Save to overleaf/ directory with specific naming
```

## Development Workflow

### Environment Setup
- Uses `.venv/` virtual environment
- Core dependencies: pandas, sklearn, statsmodels, matplotlib, seaborn
- Jupyter notebook as primary development environment

### Output Management
- **Visualizations**: Auto-saved to `overleaf/` with descriptive filenames
- **LaTeX integration**: `overleaf/main.tex` contains publication manuscript
- **Reproducibility**: Fixed random seeds (random_state=42) for consistent results

### Analysis Approach
1. **EDA phase**: Categorical analysis by season, temporal trends over years
2. **Feature selection**: ANOVA F-test ranking via `rank_features()` function
3. **Dual modeling**: scikit-learn LogisticRegression + statsmodels GLM for different insights
4. **Statistical interpretation**: Focus on odds ratios and confidence intervals

## Key Integration Points

### Data Dependencies
- Primary analysis depends on `expeditions.csv` structure (50+ columns)
- Categorical encoding assumes specific season values: Autumn, Spring, Summer, Winter
- Binary transformations require `mbrs_summited` and elevation columns

### Publication Pipeline
- Notebook generates figures → `overleaf/` directory → LaTeX manuscript
- Figure naming convention matches LaTeX references in `main.tex`
- Statistical results formatted for academic publication standards

## Mountaineering Domain Context
- **High-altitude focus**: 7,000m+ threshold represents "death zone" climbing
- **Oxygen usage**: Critical binary predictor (`is_o2_climbing`) with strong effect size
- **Route standardization**: `is_standard_rte` indicates established climbing routes vs. new routes
- **Seasonal patterns**: Spring/Autumn seasons heavily favored for success rates

This project prioritizes interpretable models over predictive accuracy to support evidence-based mountaineering decisions.