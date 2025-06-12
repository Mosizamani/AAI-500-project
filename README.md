
Introduction (Version Zero): Mount Everest remains one of the most extreme and challenging environments for human endurance, yet thousands continue to attempt its summit. Conquering the Himalaya Mountain as an extremely unique experience has its own hazards. From more than a century ago, there were a lot of people who started this adventure but some of them did not come back. Finding the factors related to the death rate can help not only the current generation but also the future adventures. Based on the research by ​(Fontanarosa et al., 2000)​, the use of supplemental oxygen has been shown to significantly reduce the risk of death among mountaineers descending from the summits of Everest and K2. Their findings highlight that climbers who forgo supplemental oxygen face substantially higher mortality rates, particularly on K2. This underscores the critical role of oxygen support in mitigating physiological stress and enhancing survival in extreme high-altitude environments. Since Mount Everest became accessible from both Nepal and Tibet, its climbing history has been shaped by changing geopolitical boundaries and international expedition dynamics. This paper ​(Huey & Salisbury, 2003)​ examines patterns of success and mortality across thousands of Everest climbers, revealing how route choice, nationality, and historical context influence mountaineering outcomes. A detailed analysis of 86 years of expedition data reveals that late summit times and symptoms such as fatigue and cognitive impairment are strong indicators of potential mortality at extreme altitudes. These findings emphasize the critical importance of timely decision-making and physical readiness in high-altitude mountaineering ​(Firth et al., 2008)​. (Westhoff et al., 2012) found that individual experience or participation in traditional (non-commercial) expeditions does not significantly improve survival odds. Instead, their study highlights that broader, collective progress—such as innovations in equipment, logistics, and shared knowledge—plays a more critical role in reducing fatalities over time. A study by ​(Anicich et al., 2015)​ revealed that cultural values related to hierarchy can both enhance and endanger expedition outcomes in the Himalayas. While hierarchical teams were more likely to reach the summit, they also experienced higher death rates—highlighting the complex trade-off between team coordination and psychological safety in high-risk group settings. ​(Gugglberger, 2018)​ explored the evolving role of women in Himalayan mountaineering, emphasizing not only their increasing participation but also the unique risks and challenges they face. The analysis sheds light on how gender dynamics intersect with survival outcomes in high-altitude expeditions. A study by ​(Huey et al., 2020)​ analyzed nearly 6,000 first-time climbers over two time periods and found that while probabilities of summiting have increased—particularly among women and older climbers—death rates have remained largely unchanged, with age emerging as a significant risk factor. The 2019 Everest Expedition, led by National Geographic and Rolex, marked the most comprehensive scientific study ever conducted on the mountain. The research revealed emerging risks to both climbers and local communities driven by environmental and human-induced changes, highlighting the urgent need to understand and mitigate evolving threats in the Everest and Khumbu region ​(Miner et al., 2020)​. ​(Krishnagopal, 2021)​ employs a novel multiscale network approach to analyze how both individual traits—such as age, gender, and experience—and expedition-wide factors influence mountaineering success. The findings highlight that climbing with familiar teammates and factors like youth and oxygen use significantly improve success rates, while expedition size and duration also play crucial roles. Mountaineering uniquely combines individual skill and endurance with the critical influence of social dynamics among climbers. Recent research ​(Krishnagopal, 2022)​ demonstrates that the structure and strength of relationships within climbing teams significantly affect cooperation levels, summit success, and even death rates, with both individual traits and expedition-wide factors playing important roles in these outcomes. The high-altitude environment presents significant health risks to mountaineers, including cold injuries such as frostbite and the potential for mortality. Recent reviews ​(Kriemler et al., 2023)​ suggest that female mountaineers may experience a lower risk of death compared to their male counterparts, though data on sex differences in frostbite remain inconclusive and warrant further investigation.  



Modeling Section 

Model Selection and Justification 
Given the binary nature of the target variable mbrs_summited (summit success: yes/no), logistic regression is a suitable modeling choice. It enables estimation of the probability of success while offering interpretability through odds ratios. Logistic regression is widely used for classification problems, especially when the goal is to understand the influence of features on binary outcomes. 
Mathematical Formulation 
Logistic regression models the log-odds of the binary response as a linear combination of the predictors: 
log(𝑝1−𝑝)=𝛽0+𝛽1𝑋1+𝛽2𝑋2+⋯+𝛽𝑘𝑋𝑘log⁡p1−p=𝛽0 +𝛽1 X1 +𝛽2 X2 +⋯+𝛽k Xk  
where: 
p is the probability of summiting, 
𝑋1,𝑋2,…,𝑋𝑘 X1 ,X2 ,…,Xk  
are the independent variables (e.g., peak, season), 
𝛽0,…,𝛽𝑘 𝛽0,…,𝛽k  
are model parameters.  
To interpret results, we exponentiate the coefficients: 
𝑂𝑑𝑑𝑠𝑅𝑎𝑡𝑖𝑜=𝑒𝛽𝑖 OddsRatio=e𝛽i
An odds ratio >1 implies an increase in summit probability when the predictor increases. 
Data Preparation 
To ensure data quality and model relevance, expeditions reaching above 7000 meters were selected to focus on high-altitude scenarios. Missing values were removed, and the target variable mbrs_summited was binarized for classification. Categorical features were encoded into numeric form to support regression and classification modeling. 
Filtering: Only expeditions with max_elev_reached > 7000 meters were retained to focus on high-risk cases. 
Cleaning: Rows with missing data were removed. 
Target Binarization: mbrs_summited was converted to 0 or 1. 
Categorical Encoding: Categorical features like season, route name, and leaders were numerically encoded to make them usable in the model. 
Model Training and Evaluation 
The dataset was split into training and test sets (80/20), and a logistic regression model was trained with max_iter=1000 to ensure convergence. Model performance was assessed using accuracy, confusion matrix, classification report, and ROC-AUC metrics. Additionally, a 0.7 probability threshold was applied to improve confidence in binary classification decisions. 
Data was split into 80% training and 20% test sets. 
Logistic regression was trained with max_iter=1000 to ensure convergence. 
Model performance was evaluated using: 
Accuracy: Proportion of correct predictions. 
Confusion Matrix: Summarized true/false positives/negatives. 
Classification Report: Precision, recall, F1-score. 
ROC Curve & AUC: Visualized and quantified classifier performance. 
A threshold of 0.7 was tested for binary classification based on predicted probabilities to increase decision confidence. 
Key Results 
The logistic regression model achieved an accuracy of 0.78, indicating reliable performance. Key predictors of summit success included is_standard_rte, is_o2_climbing, and season, all with odds ratios greater than 1—suggesting a positive impact on the likelihood of summiting. The AUC score of 0.79 further demonstrates the model’s strong ability to distinguish between successful and unsuccessful expeditions. 
The model's accuracy was 0.78. 
The most influential features included: 
is_standard_rte, is_o2_climbing, season 
These features had odds ratios >1, indicating they positively influence summit success. 
AUC Score: 0.79, showing strong model discrimination. 
Limitations 
While the model offers useful insights, several limitations should be considered. The simplification of categorical variables through encoding may overlook important contextual relationships. Additionally, the model does not account for feature interactions or non-linear patterns, and relying on a single train-test split may affect the reliability of evaluation results. 
Categorical variables were simplified via encoding, possibly missing nuance. 
No feature interaction terms or non-linearities were modeled. 
Single train-test split may lead to overfitting or optimistic evaluation. 
Logistic regression assumes linearity in log-odds, which may not fully capture real-world complexity. 
Recommendations 
To enhance model reliability and performance, several improvements have been suggested. Implementing k-fold cross-validation can provide a more robust evaluation by reducing variance due to data splitting. Additionally, exploring advanced models like Random Forest or XGBoost, incorporating interaction terms, and integrating domain-specific features can help capture complex patterns and improve predictive accuracy. 
Use k-fold cross-validation to ensure robust evaluation. 
Consider tree-based models (e.g., Random Forest, XGBoost) for better performance on nonlinear relationships. 
Add interaction terms or use polynomial logistic regression to capture more complex behavior. 
Explore domain-specific features (e.g., climber experience level, weather data) for future models. 
Discussion 
This logistic regression model offers interpretable insight into expedition success factors. Supplemental oxygen use, standard route selection, and commercial organization positively correlate with summit outcomes. These findings can inform mountaineers and organizers in risk planning and resource allocation. However, careful interpretation is required due to modeling simplifications and data limitations. Further analysis using advanced models and enriched data would enhance prediction and decision-making reliability. 
 

