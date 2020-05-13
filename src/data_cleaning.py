import pandas as pd
import numpy as np
from itertools import chain


def data_cleaning(infile, outfile):
    df_raw = pd.read_csv(infile)
    df_clean = df_raw.copy()

    # make lowercase
    df_clean['text'] = df_clean['text'].str.lower()

    # drop retweets
    df_clean = df_clean[~df_clean['text'].str.contains(r"[\"\”\“]\@\S+\:", regex=True)]

    # get df of all tags in a tweet
    tags = df_clean['text'].str.extractall(r'(@\w+)')

    # remove links
    df_clean['text'] = df_clean['text'].str.replace(r"https*://\S+", "", regex=True)

    # remove hashtags
    df_clean['text'] = df_clean['text'].str.replace(r"\#\S+", "", regex=True)

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

        
    # replace tagged airlines

    airlines_dict = {
        "Virgin America": {
            "virginamerica",
            "virgin",
            "virginatlantic",
            "virginmedia"
        },
        "United": {
            "united",
            "united_airline",
            "unitedairlines",
            "unitedappeals",
            "unitedflyerhd"
            
        },
        "Southwest": {
            "southwestair",
            "southwest"
        },
        "Delta": {
            "deltaassist",
            "delta",
            "deltapoints"
        },
        "US Airways": {
            "usairways",
            "usair",
            "usairwayscenter",
            "usairwaysmobile",
            "usairwis"
        },
        "American": {
            "americanair",
            "americanairbr",
            "americanairlines",
            "americanairlnes"
        },
        "Other Airlines": {
            "jetblue",
            "aircanada",
            "british_airways",
            "continentalair1",
            "flyfrontier",
            "jetbluecheeps",
            "spiritairlines",
            "spiritairpr",
            "westjet"
        }
    }

    users_of_interest = set(chain.from_iterable(airlines_dict.values()))

    for index, row in tags.iterrows():
        row_idx = index[0]
        row_airline = df_clean['airline'][row_idx].strip()
        tagged_user = row[0].replace("@", "")
        if tagged_user in users_of_interest:
            for airline in airlines_dict:
                if tagged_user in airlines_dict[airline]:
                    if airline == row_airline:
                        df_clean.loc[row_idx, 'text'] = df_clean.loc[row_idx, 'text'].replace(tagged_user, "airlineaccount")
                    else:
                        df_clean.loc[row_idx, 'text'] = df_clean.loc[row_idx, 'text'].replace(tagged_user, "otherairlineaccount")
        else:
            df_clean.loc[row_idx, 'text'] = df_clean.loc[row_idx, 'text'].replace(tagged_user, "useraccount")

    # remove trailing/leading whitespace
    df_clean['text'] = df_clean['text'].apply(str.strip)

    df_clean.to_csv(outfile)
    return

if __name__ == "__main__":
    data_cleaning("data/Tweets.csv", "data/Clean_Tweets.csv")