from joblib import dump, load
from statistics import mode, StatisticsError
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import nltk

from src.data_cleaning import data_cleaning
from src.identify_users_of_interest import remove_users_of_interest
from src.helpers import print_model_metrics, wordnet_lemmetize_tokenize
from src.vader_analysis import VaderAnalysis
from src.d2v_custom import CustomDoc2Vec
from src.evaluate_models import ensemble_models_probas, format_probas

def normalize_proba(probas_list):
    out = []
    for prob in probas_list:
        total = sum(prob.values())
        out_dict = {}
        for key in prob:
            out_dict[key] = prob[key] / total
        out.append(out_dict)
    return out

def predict_one(text):
    nb = load("models/naivebayes.joblib")
    rf = load("models/randomforest.joblib")
    d2v_rf = load("models/d2v_randomforest.joblib")
    vader = VaderAnalysis()

    clean_text = data_cleaning(text)
    
    nb_proba = format_probas(nb.classes_, nb.predict_proba([clean_text]))
    rf_proba = format_probas(rf.classes_, rf.predict_proba([clean_text]))
    d2v_rf_proba = format_probas(d2v_rf.classes_, d2v_rf.predict_proba([clean_text]))
    vader_proba = vader.predict_proba([clean_text])
    ensemble_preds, ensemble_probas = ensemble_models_probas(rf_proba, nb_proba, vader_proba, d2v_rf_proba)

    normal_ensemble_proba = normalize_proba(ensemble_probas)

    outdict = {
        "Prediction": [
            ensemble_preds[0],
            max(nb_proba[0], key=nb_proba[0].get),
            max(rf_proba[0], key=rf_proba[0].get),
            max(d2v_rf_proba[0], key=d2v_rf_proba[0].get),
            max(vader_proba[0], key=vader_proba[0].get),
        ],
        "Negative Probability": [
            round(normal_ensemble_proba[0]['negative'], 2),
            round(nb_proba[0]['negative'], 2),
            round(rf_proba[0]['negative'], 2),
            round(d2v_rf_proba[0]['negative'], 2),
            round(vader_proba[0]['negative'], 2),
        ],
        "Neutral Probability": [
            round(normal_ensemble_proba[0]['neutral'], 2),
            round(nb_proba[0]['neutral'], 2),
            round(rf_proba[0]['neutral'], 2),
            round(d2v_rf_proba[0]['neutral'], 2),
            round(vader_proba[0]['neutral'], 2),
        ],
        "Positive Probability": [
            round(normal_ensemble_proba[0]['positive'], 2),
            round(nb_proba[0]['positive'], 2),
            round(rf_proba[0]['positive'], 2),
            round(d2v_rf_proba[0]['positive'], 2),
            round(vader_proba[0]['positive'], 2),
        ]
    }

    return pd.DataFrame(outdict, index=[
            "Ensemble",
            "Naive Bayes",
            "Random Forest (TF-IDF)",
            "Random Forest (Doc2Vec)",
            "VADER"
        ])

if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('vader_lexicon')  
    print(predict_one("@united thanks for the terrible service"))