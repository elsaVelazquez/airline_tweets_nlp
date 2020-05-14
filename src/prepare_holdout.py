from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("data/Tweets.csv")
training, holdout = train_test_split(data, shuffle=True, random_state=10)

training.to_csv("data/T_Tweets.csv")
holdout.to_csv("data/Holdout_Tweets.csv")