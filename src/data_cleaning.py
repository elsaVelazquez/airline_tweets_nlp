import pandas as pd
import numpy as np
from itertools import chain
from src.helpers import remove_hashtags, clean_whitespace, pad_emojis, remove_punctuation


def data_cleaning(infile, outfile=None):
    # Load data
    if outfile != None:
        df_raw = pd.read_csv(infile)
        df_clean = df_raw.copy()
    else:
        df_clean = pd.DataFrame({"text": [infile]})

    # make lowercase
    df_clean['text'] = df_clean['text'].str.lower()

    # remove links and urls
    df_clean['text'] = df_clean['text'].str.replace(
                                                    r"https*://\S+",
                                                    "",
                                                    regex=True
                                                )

    # remove hashtags
    df_clean['text'] = df_clean['text'].apply(
                                lambda x: remove_hashtags(x, keep_text=True)
                            )

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
        "”": "\"",
        "“": "\""
    }

    for char in remove_char_list:
        df_clean['text'] = df_clean['text'].str.replace(char, "")

    for char in replace_char_dict:
        df_clean['text'] = df_clean['text'].str.replace(
                                                    char,
                                                    replace_char_dict[char]
                                                )

    df_clean['text'] = df_clean['text'].apply(remove_punctuation)
    df_clean['text'] = df_clean['text'].apply(pad_emojis)

    # remove trailing/leading whitespace
    df_clean['text'] = df_clean['text'].apply(clean_whitespace)
    if outfile != None:
        df_clean.to_csv(outfile)
        return
    else:
        return df_clean['text'][0]


if __name__ == "__main__":
    data_cleaning("data/T_Tweets.csv", "data/Clean_T_Tweets.csv")
