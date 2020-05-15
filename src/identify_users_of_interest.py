from itertools import chain
import pandas as pd


def remove_users_of_interest(df, text_col, user_col):
    '''
    Standardize Twitter users in a specified columns

    Ex. Will transform
    |      | airline               | text                                                                            |
    |--  :|:--------------  |:-----------------------------------------------------|
    |  0   | Virgin America      | @virginamerica I love your airline. @jack you should check them out! |
    |  1   | Southwest            | @southwestair I hate your airline! @united is way better!              |

    to:
    |      | airline               | text                                                                                     |
    |--  :|:--------------  |:-----------------------------------------------------------|
    |  0   | Virgin America      | @airlineaccount I love your airline. @useraccount you should check them out! |
    |  1   | Southwest            | @airlineaccount I hate your airline! @otherairlineaccount is way better!      |
    '''
    # get df of all tags in a tweet
    tags = df[text_col].str.extractall(r'(@\w+)')

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
        row_airline = df[user_col][row_idx].strip()
        tagged_user = row[0].replace("@", "")
        if tagged_user in users_of_interest:
            for airline in airlines_dict:
                if tagged_user in airlines_dict[airline]:
                    if airline == row_airline:
                        df.loc[row_idx, text_col] = df.loc[row_idx, text_col].replace(tagged_user, "airlineaccount")
                    else:
                        df.loc[row_idx, text_col] = df.loc[row_idx, text_col].replace(tagged_user, "otherairlineaccount")
        else:
            df.loc[row_idx, text_col] = df.loc[row_idx, text_col].replace(tagged_user, "useraccount")
    return df


if __name__ == '__main__':
    df = pd.read_csv("data/Clean_T_Tweets.csv")
    df = remove_users_of_interest(df, 'text', 'airline')
    df.to_csv("data/Clean_T_Tweets_wo_Users.csv")
