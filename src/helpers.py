import sklearn.feature_extraction.text
import nltk
from string import punctuation
import numpy as np
import re


sw = [
    'my', 'is', 'in', 'it', 'no', 'of', 'not', 'your', 'me', 'hour', 'have', 'wa', 'that', 'to', 'the', 'i', 'you', 'u', 'on', 'a', 'do', 'for', 'at', 'so', 'and', 'be', 'now', 'with', 'just', 'get', 'our', 'we', 'an', 'are', 'this', 'but', 'will', 'fleek', 'im'
]
additional_lemmatizate_dict = {
    "cancelled": "cancel",
    "cancellation": "cancel",
    "cancellations": "cancel",
    "delays": "delay",
    "delayed": "delay",
    "baggage": "bag",
    "bags": "bag",
    "luggage": "bag",
    "dms": "dm"
}

def create_stop_words(additional_stopwords=None):
    if additional_stopwords:
        return sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.union(additional_stopwords)
    else:
        return sklearn.feature_extraction.text.ENGLISH_STOP_WORDS

def remove_punctuation(string, punc=punctuation):
    # remove given punctuation marks from a string
    for character in punc:
        string = string.replace(character,'')
    return string

def remove_hashtags(string, keep_text=False):
    if keep_text:
        return re.sub(r"\#([^\s\#]+)", r"\1", string) 
    else:
        return re.sub(r"\#\S+", "", string)

def remove_tagged_users(string):
    return re.sub(r"\@\S+", "", string)

def remove_line_breaks(string):
     return re.sub(r"\n", " ", string)
    
def clean_whitespace(string):
     return re.sub(r"\s+", " ", string.strip())

def clean_df_column(df, col):
    df[col] = df[col].apply(remove_hashtags)
    df[col] = df[col].apply(remove_tagged_users)
    df[col] = df[col].apply(remove_line_breaks)
    df[col] = df[col].apply(clean_whitespace)
    df[col] = df[col].str.strip()
    return df

def define_axis_style(ax, title, x_label, y_label, legend=False):
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xlabel(x_label, fontsize=16)
    ax.tick_params(labelsize=14)
    if legend:
        ax.legend(fontsize=16)
    return