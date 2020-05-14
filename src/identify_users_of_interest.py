from itertools import chain
import pandas as pd

# I did not find this function to improve predictions

def remove_users_of_interest(df):
    # get df of all tags in a tweet
    tags = df['text'].str.extractall(r'(@\w+)')

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
        row_airline = df['airline'][row_idx].strip()
        tagged_user = row[0].replace("@", "")
        if tagged_user in users_of_interest:
            for airline in airlines_dict:
                if tagged_user in airlines_dict[airline]:
                    if airline == row_airline:
                        df.loc[row_idx, 'text'] = df.loc[row_idx, 'text'].replace(tagged_user, "airlineaccount")
                    else:
                        df.loc[row_idx, 'text'] = df.loc[row_idx, 'text'].replace(tagged_user, "otherairlineaccount")
        else:
            df.loc[row_idx, 'text'] = df.loc[row_idx, 'text'].replace(tagged_user, "useraccount")
    return df

if __name__ == '__main__':
    df = pd.read_csv("data/Clean_T_Tweets.csv")
    df = remove_users_of_interest(df)
    df.to_csv("data/Clean_T_Tweets_wo_Users.csv")