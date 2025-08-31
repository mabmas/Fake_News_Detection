Technical Approach:

The system uses Sentence-BERT for deep semantic text understanding, which captures contextual meaning more effectively than methods like TF-IDF. Predictions are made by a soft-voting 
ensemble of XGBoost and a Support Vector Machine (SVM). To ensure fairness and reduce bias from imbalanced data, the model integrates Synthetic Minority Oversampling Technique (SMOTE). Performance 
is further enhanced through automated hyperparameter tuning using Dispersive Fly Optimization (DFO). The core technologies include Python, Scikit-learn, XGBoost, and SBERT.

Performance:

On the LIAR dataset, the system achieved 65.46% accuracy. This result outperforms traditional standalone classifiers, including Decision Tree (55.2%) and Random Forest (59.5%). 
The model demonstrated balanced performance with F1-scores of 0.64 for Fake news and 0.67 for Real news


