import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
from helpers import clean_df_column, print_model_metrics


def demo_vader(sentence):
    '''
    Demo VADER sentiment analysis
    '''
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))


class VaderAnalysis():
    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()

    def predict(self, X):
        y_preds = []
        threshold = 0.001
        for text in X:
            score = self.analyser.polarity_scores(text)
            if abs(score['compound']) >= threshold:
                if score['compound'] < 0:
                    y_preds.append("negative")
                elif score['compound'] > 0:
                    y_preds.append('positive')
            else:
                y_preds.append("neutral")
        return y_preds

    def predict_proba(self, X):
        '''
        Predict positive, neutral, negative sentiment probabilities
        X : array of text entries
        '''
        y_preds = []
        for text in X:
            score = self.analyser.polarity_scores(text)
            del score['compound']
            # rename keys to line up with target names
            score["negative"] = score.pop("neg")
            score["positive"] = score.pop("pos")
            score["neutral"] = score.pop("neu")
            y_preds.append(score)
        return y_preds


if __name__ == '__main__':
    nltk.download('vader_lexicon')

    # Load data
    data = pd.read_csv("data/Clean_T_Tweets_wo_Users.csv")

    # Clean columns
    data = clean_df_column(data, 'text')

    vader = VaderAnalysis()

    # Make predictions
    y_preds = vader.predict(data['text'])

    y_test = data['airline_sentiment']

    # Evaluate
    print_model_metrics(y_test, y_preds)
