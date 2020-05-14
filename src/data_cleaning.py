import pandas as pd
import numpy as np
from itertools import chain
from helpers import remove_hashtags, clean_whitespace

def data_cleaning(infile, outfile):
    df_raw = pd.read_csv(infile)
    df_clean = df_raw.copy()

    # make lowercase
    df_clean['text'] = df_clean['text'].str.lower()

    # remove links and urls
    df_clean['text'] = df_clean['text'].str.replace(r"https*://\S+", "", regex=True)

    # remove hashtags
    df_clean['text'] = df_clean['text'].apply(lambda x: remove_hashtags(x, keep_text=True))

    # Remove weird characters
    remove_char_list = [
        "⤴",
        "⤵",
        u"\u0361",
        u"\u035D",
        u"\u035C"
    ]
    replace_char_dict = {
        "’": "'",
        "‘": "'",
        "”": r"\"",
        "“": r"\""
    }

    for char in remove_char_list:
        df_clean['text'] = df_clean['text'].str.replace(char, "")

    for char in replace_char_dict:
        df_clean['text'] = df_clean['text'].str.replace(char, replace_char_dict[char])    

    # remove trailing/leading whitespace
    df_clean['text'] = df_clean['text'].apply(clean_whitespace)

    df_clean.to_csv(outfile)
    return

if __name__ == "__main__":
    data_cleaning("data/T_Tweets.csv", "data/Clean_T_Tweets.csv")
    data_cleaning("data/Holdout_Tweets.csv", "data/Clean_Holdout_Tweets.csv")