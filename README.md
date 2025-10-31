Himalayan Expedition Success Predictor

## Quick Start

### Using the Improved Analysis Module
```bash
# Install dependencies
pip install -r requirements.txt

# Run the improved analysis with cross-validation
python example_analysis.py

# Or use the modular functions in your own code
python -c "from analysis_utils import load_and_analyze_expeditions; load_and_analyze_expeditions()"

# Run tests
python -m pytest test_analysis.py -v
```

### Using the Original Jupyter Notebook
Open `EDA_for_Himalayan_Summit_Success_glm.ipynb` in Jupyter Lab/Notebook for the original analysis.

## Project Overview

1. Overview
Mount Everest and other Himalayan peaks represent the pinnacle of human endurance, drawing thousands of climbers despite the extreme risks. For over a century, these mountains have been the stage for incredible triumphs and devastating tragedies. This project aims to analyze historical expedition data to identify the key factors that contribute to summit success. By understanding the variables that influence outcomes, we can provide valuable insights for climbers and expedition organizers, potentially enhancing safety and success rates for future adventurers.

2. Background Research
The dangers of high-altitude mountaineering are well-documented. Decades of research have explored various facets of climbing in the Himalayas, providing a foundation for this analysis:

Supplemental Oxygen: The use of supplemental oxygen is a critical factor in mitigating physiological stress and has been shown to significantly reduce mortality rates, particularly during descents from the highest peaks like Everest and K2 (Fontanarosa et al., 2000).

Success and Mortality Patterns: Studies have examined how route choice, nationality, geopolitical factors, late summit times, and symptoms of fatigue are strong indicators of expedition outcomes (Huey & Salisbury, 2003; Firth et al., 2008).

Experience vs. Collective Progress: Interestingly, individual experience does not always correlate with improved survival odds. Instead, collective advancements in equipment, logistics, and shared knowledge have played a more significant role in reducing fatalities over time (Westhoff et al., 2012).

Team and Social Dynamics: The structure of a climbing team is crucial. Hierarchical teams may be more likely to summit but can also face higher death rates, highlighting a trade-off between coordination and psychological safety (Anicich et al., 2015). Furthermore, climbing with familiar teammates has been shown to improve success rates (Krishnagopal, 2021; Krishnagopal, 2022).

Demographics and Risk: Recent analyses show that while summit probabilities have increased—especially for women and older climbers—mortality rates have remained stable, with age being a significant risk factor (Huey et al., 2020). Gender also plays a role, with some evidence suggesting female mountaineers may have a lower risk of death compared to males (Gugglberger, 2018; Kriemler et al., 2023).

Environmental Changes: The Everest region faces new risks from environmental and human-induced changes, underscoring the need for ongoing scientific research to protect climbers and local communities (Miner et al., 2020).

3. Modeling and Analysis
This project uses a logistic regression model to predict the likelihood of an expedition successfully summiting.

3.1. Model Selection: Logistic Regression
Given the binary nature of the outcome (summit: yes/no), logistic regression was chosen for its ability to model probabilities and provide interpretable results through odds ratios. It allows us to understand how individual features influence the chance of success.

The model uses the following mathematical formulation, which describes the log-odds of the outcome as a linear combination of predictor variables:

log(
1−p
p
​
 )=β
0
​
 +β
1
​
 X
1
​
 +β
2
​
 X
2
​
 +⋯+β
k
​
 X
k
​

Where:

p is the probability of a successful summit.

X
1
​
 ,X
2
​
 ,…,X
k
​
  are the independent variables (e.g., season, use of oxygen).

β
0
​
 ,…,β
k
​
  are the model parameters.

The impact of each predictor is interpreted using the odds ratio, calculated as e
β
i
​

 . An odds ratio greater than 1 implies that an increase in the predictor variable increases the probability of summiting.

3.2. Data Preparation
To ensure data quality and model relevance, the following steps were taken:

Filtering: The dataset was filtered to include only expeditions that reached an altitude above 7,000 meters, focusing the analysis on high-altitude, high-risk scenarios.

Cleaning: Rows with missing values were removed to maintain data integrity.

Target Binarization: The target variable, mbrs_summited, was converted into a binary format (1 for success, 0 for failure).

Feature Encoding: Categorical features such as season, route name, and leaders were numerically encoded to be used in the regression model.

3.3. Model Training and Evaluation
The dataset was split into an 80% training set and a 20% test set.

A logistic regression model was trained with max_iter=1000 to ensure the algorithm had sufficient iterations to converge.

Performance was assessed using standard classification metrics:

Accuracy: The proportion of correctly classified predictions.

Confusion Matrix: A summary of true positives, true negatives, false positives, and false negatives.

Classification Report: Detailed precision, recall, and F1-scores for each class.

ROC Curve & AUC: A measure of the model's ability to distinguish between classes.

4. Key Results
The logistic regression model demonstrated strong predictive performance in identifying factors related to summit success.

Accuracy: 78%

AUC Score: 0.79

The model's Area Under the Curve (AUC) score of 0.79 indicates a good ability to distinguish between successful and unsuccessful expeditions.

The most influential predictors of summit success were:

is_standard_rte (climbing a standard route)

is_o2_climbing (using supplemental oxygen)

season

All these features had odds ratios greater than 1, confirming that climbing on a standard route, using supplemental oxygen, and climbing in a favorable season positively and significantly influence the likelihood of a successful summit.

5. Limitations
While the model provides valuable insights, it is important to acknowledge its limitations:

Data Simplification: Encoding categorical variables simplifies complex information (e.g., specific routes), potentially overlooking important contextual nuances.

Linearity Assumption: Logistic regression assumes a linear relationship in the log-odds, which may not fully capture the complex, non-linear interactions present in real-world mountaineering.

Feature Interactions: The current model does not account for potential interactions between features (e.g., how the effect of experience might vary with route difficulty).

Single Data Split: The evaluation is based on a single train-test split, which may lead to results that are overly optimistic or not generalizable.

6. Recommendations for Future Work
To build upon this analysis, the following improvements are recommended:

**✅ COMPLETED IMPROVEMENTS:**

**Cross-Validation Implementation**: Replaced single train-test split with 5-fold stratified cross-validation to obtain more robust and reliable model performance estimates. See `analysis_utils.py` for implementation.

**Modular Code Organization**: Extracted reusable functions from the notebook into `analysis_utils.py` module with proper data validation, error handling, and consistent API design.

**Automated Testing**: Added comprehensive unit tests in `test_analysis.py` to ensure code reliability and data integrity validation.

**Development Environment**: Created `requirements.txt` for reproducible dependency management and GitHub Actions workflow for automated testing.

**REMAINING FUTURE WORK:**

Explore Advanced Models: Employ tree-based models like Random Forest or XGBoost, which can capture non-linear relationships and feature interactions automatically.

Feature Engineering: Incorporate interaction terms or polynomial features into the logistic regression model to capture more complex behaviors.

Enrich the Dataset: Integrate additional domain-specific features, such as detailed weather data, climber experience levels, or team composition metrics, to enhance predictive accuracy.

7. Discussion
This analysis successfully developed an interpretable model that aligns with established mountaineering knowledge. The findings confirm that the use of supplemental oxygen, adherence to standard routes, and climbing within organized expeditions are strongly correlated with summit success. These insights can be directly applied by mountaineers and organizers to inform risk management, strategic planning, and resource allocation. While the model has limitations, it serves as a strong foundation for future, more sophisticated analyses that could further enhance safety and decision-making in the high-stakes environment of Himalayan climbing.

8. References
Anicich, E. M., Swaab, R. I., & Galinsky, A. D. (2015). Hierarchical cultural values predict success and mortality in high-stakes teams. Psychological Science, 26(2), 168-176.

Firth, P. G., Zheng, H., Windsor, J. S., Sutherland, A. I., Imray, C. H., Moore, G. W. K., & Roberts, G. W. K. (2008). Mortality on Mount Everest, 1921-2006: Descriptive study. BMJ, 337, a2654.

Fontanarosa, P. B., et al. (2000). Supplemental Oxygen and Risk of Death on Everest and K2. JAMA.

Gugglberger, L. (2018). Women in Himalayan mountaineering: A gendered analysis of risk and success. Journal of Sport and Social Issues, 42(3), 197-217.

Huey, R. B., & Salisbury, R. (2003). The geography of mortality on Mount Everest. The Geographical Journal, 169(1), 1-13.

Huey, R. B., et al. (2020). Rates of success and death of climbers on Mount Everest. PLOS ONE, 15(10), e0239564.

Kriemler, S., et al. (2023). Sex differences in mountaineering: A narrative review on mortality and cold injuries. High Altitude Medicine & Biology.

Krishnagopal, S. (2021). Multiscale network analysis of mountaineering success. Scientific Reports, 11(1), 1-13.

Krishnagopal, S. (2022). Social dynamics in mountaineering: The role of individual traits and expedition-wide factors. Applied Network Science, 7(1), 1-21.

Miner, A. P., et al. (2020). The 2019 National Geographic and Rolex Perpetual Planet Everest Expedition. Science, 369(6511), 1561-1563.

Westhoff, T. H., et al. (2012). Does individual experience or collective progress reduce mortality in Himalayan mountaineering? Swiss Medical Weekly, 142, w13615.