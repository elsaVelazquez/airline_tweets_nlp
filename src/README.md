# Order of Operations

To recreate this analysis, run scripts in the following order:

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Download Data
Download dataset from https://www.kaggle.com/crowdflower/twitter-airline-sentiment and place in `/data/` directory (located in the root directory of the project). The dataset should be called `Tweets.csv`.

## Prepare Data

Separate holdout data:
```bash
python3 src/prepare_holdout.py
```

Clean training data:
```bash
python3 src/data_cleaning.py
```

Standardize Twitter users:
```
python3 src/identify_users_of_interest.py
```

## Explore Data (Optional)

EDA
```
python3 src/eda.py
```

Get PCA plots:
```
python3 src/pca.py
```

```
python3 src/pca_animation.py
```

## Create Models

Create NaiveBayes Model:
```
python3 src/naive_bayes.py
```

Create RandomForest Model:
```
python3 src/randomforest.py
```

## Evaluate Models
```
python3 src/evaluate_models.py
```