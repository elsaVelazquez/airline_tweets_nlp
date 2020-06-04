from joblib import dump, load
from statistics import mode, StatisticsError
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data_cleaning import data_cleaning
from src.identify_users_of_interest import remove_users_of_interest
from src.helpers import print_model_metrics, wordnet_lemmetize_tokenize
from src.vader_analysis import VaderAnalysis
from src.d2v_custom import CustomDoc2Vec


def ensemble_models_voting(rf_preds, nb_preds, vader_preds, d2v_rf_preds):
    '''
    Assemble predictions from RandomForest, NaiveBayes, and VADER analysis.
    If no majority can be reached, ties get determined by RandomForest
    '''
    final_preds = []
    for preds in zip(rf_preds, nb_preds, vader_preds, d2v_rf_preds):
        try:
            final_pred = mode(preds)
        except StatisticsError:
            # tie goes to d2v random forest
            final_pred = preds[3]
        final_preds.append(final_pred)
    return final_preds


def ensemble_models_probas(rf_probas, nb_probas, vader_probas, d2v_rf_probas, weights=[1, 0.2, 0.2, 0.6]):
    '''
    Assemble predictions in the form of probabilities
    from RandomForest, NaiveBayes, and VADER analysis.
    '''
    rf_weight = weights[0]
    nb_weight = weights[1]
    vader_weight = weights[2]
    d2v_rf_weight = weights[3]
    final_preds = []
    final_probas = []
    for preds in zip(rf_probas, nb_probas, vader_probas, d2v_rf_probas):
        pred_dict = {"negative": 0, "positive": 0, "neutral": 0}
        for key in pred_dict:
            pred_dict[key] += preds[0][key] * rf_weight
            pred_dict[key] += preds[1][key] * nb_weight
            pred_dict[key] += preds[2][key] * vader_weight
            pred_dict[key] += preds[3][key] * d2v_rf_weight
        final_preds.append(max(pred_dict, key=pred_dict.get))
        final_probas.append(pred_dict)
    return final_preds, final_probas


def format_probas(labels, probas):
    '''
    Format SKLearn predict_proba outputs
    to be compared with VADER sentiment analysis
    '''
    probas_dicts = []
    for pred in probas:
        probas_dicts.append(dict(zip(labels, pred)))
    return probas_dicts


if __name__ == '__main__':
    # Load models
    rf_pipeline = load('models/randomforest.joblib')
    nb_pipeline = load('models/naivebayes.joblib')
    d2v_pipeline = load('models/d2v_randomforest.joblib')

    # Load and clean holdout data
    data_cleaning("data/Holdout_Tweets.csv", "data/Clean_Holdout_Tweets.csv")
    holdout_df = pd.read_csv("data/Clean_Holdout_Tweets.csv")
    holdout_df = remove_users_of_interest(holdout_df, 'text', 'airline')
    holdout_df.to_csv("data/Clean_Holdout_Tweets_wo_Users.csv")

    X_holdout = holdout_df['text']
    y_holdout = holdout_df['airline_sentiment']

    # Predict all values as majority class (negative) and evaluate
    print("Baseline Metrics:")
    print_model_metrics(y_holdout, ["negative"] * len(y_holdout))

    print("\n\n")

    # Predict just using NaiveBayes and print model evaluations
    y_preds_nb = nb_pipeline.predict(X_holdout)
    print("Naive Bayes Metrics:")
    print_model_metrics(y_holdout, y_preds_nb)

    print("\n\n")

    # Predict just using Random Forest and print model evaluations
    y_preds_rf = rf_pipeline.predict(X_holdout)
    print("Random Forest Metrics:")
    print_model_metrics(y_holdout, y_preds_rf)

    print("\n\n")

    # Predict just using VADER and print model evaluations
    vader = VaderAnalysis()
    y_preds_vader = vader.predict(X_holdout)
    print("VADER Metrics:")
    print_model_metrics(y_holdout, y_preds_vader)

    print("\n\n")

    y_preds_d2v_rf = d2v_pipeline.predict(X_holdout)

    print("Doc2Vec Random Forest Metrics:")

    print_model_metrics(y_holdout, y_preds_d2v_rf)

    print("\n\n")

    # Predict just using voting method and print evaluations
    y_preds_voting = ensemble_models_voting(y_preds_rf, y_preds_nb, y_preds_vader, y_preds_d2v_rf)
    print("Ensemble Voting Metrics:")
    print_model_metrics(y_holdout, y_preds_voting)

    # Format probabilities to be compared with VADER
    # Get RF probabilities
    y_probas_rf = rf_pipeline.predict_proba(X_holdout)
    labels = rf_pipeline.classes_
    rf_fmt_probas = format_probas(labels, y_probas_rf)

    # Get NB probabilities
    y_probas_nb = nb_pipeline.predict_proba(X_holdout)
    labels = nb_pipeline.classes_
    nb_fmt_probas = format_probas(labels, y_probas_nb)

    # Get VADER probabilities
    y_probas_vader = vader.predict_proba(X_holdout)

    # Get D2V RF probabilities
    y_probas_d2v_rf = d2v_pipeline.predict_proba(X_holdout)
    labels = d2v_pipeline.classes_
    d2v_rf_fmt_probas = format_probas(labels, y_probas_d2v_rf)

    # Predict just using all model probabilities and print evaluations


    print("Ensemble Probas Metrics:")
    y_preds_probas, _ = ensemble_models_probas(
                        rf_fmt_probas,
                        nb_fmt_probas,
                        y_probas_vader,
                        d2v_rf_fmt_probas,
                        weights=[1, 0.2, 0.2, 0.6]
                    )
    print_model_metrics(y_holdout, y_preds_probas)