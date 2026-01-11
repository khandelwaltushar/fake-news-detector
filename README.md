## Problem Statement
Misinformation spreads rapidly through online news platforms and social media, influencing public opinion and decision-making.
The goal of this project is to build an AI-powered system that analyzes news headlines or full articles and estimates the likelihood that the content is real or fake, along with a calibrated credibility score and an explanation of the model’s reasoning.


## Success Criteria

Functional Success
User can input a headline or article
System returns:
Fake / Real label
Credibility score (0–100)
Explanation text
Model confidence

ML Success
Beats TF-IDF + Logistic Regression baseline
Precision for FAKE class ≥ recall trade-off (We want fewer FPs as much as possible)
Stable probabilities (calibrated)

Engineering Success
Inference < 500 ms on CPU
Reproducible training
Modular codebase


## Scope
Binary classification (REAL / FAKE)
One primary dataset (Fake & Real News – Kaggle)
One transformer model (DistilBERT)
Streamlit UI
Basic explanation




We start with the Fake and Real News dataset from Kaggle due to its balanced labels and clean article-level text, and later validate generalization using the LIAR dataset.





