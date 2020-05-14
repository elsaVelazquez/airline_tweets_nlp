import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix
import re
from helpers import clean_df_column, print_model_metrics


def demo_vader(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

def vader_predict_proba(X):
    analyser = SentimentIntensityAnalyzer()
    y_preds = []
    for text in X:
        score = analyser.polarity_scores(text)
        del score['compound']
        score["negative"] = score.pop("neg")
        score["positive"] = score.pop("pos")
        score["neutral"] = score.pop("neu")
        y_preds.append(score)
    return y_preds

def vader_predict(X):
    analyser = SentimentIntensityAnalyzer()
    y_preds = []
    threshold = 0.001
    for text in X:
        score = analyser.polarity_scores(text)
        if abs(score['compound']) >= threshold:
            if score['compound'] < 0:
                y_preds.append("negative")
            elif score['compound'] > 0:
                y_preds.append('positive')
        else:
            y_preds.append("neutral")
    return y_preds

if __name__ == '__main__':
    nltk.download('vader_lexicon')

    data = pd.read_csv("data/Clean_T_Tweets_wo_Users.csv")

    data = clean_df_column(data, 'text')

    y_preds = vader_predict(data['text'])

    y_test = data['airline_sentiment']

    print_model_metrics(y_test, y_preds)
    
    demo_vader("I hate you!")