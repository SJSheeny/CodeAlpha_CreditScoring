# Credit Scoring Model

This is my Task 1 project for the CodeAlpha machine learning internship. The idea is to predict whether someone applying for a loan is likely to default, using their financial history: income, debts, how often they've paid late, and so on.

## A note about the data

I couldn't get hold of a real bank dataset. I tried the Kaggle "Give Me Some Credit" data, but I couldn't get the download to work, so I generated my own instead. `generate_data.py` creates 30,000 made-up applicants and saves them to `credit_data.csv`. I used the same column names as the Kaggle file, so a real dataset could be dropped in later. About 6.6% of the applicants default, and around 20% of the incomes are missing, which is roughly what real lending data looks like.

Nobody in this file is real, and the patterns the models find are patterns I built in myself. So the results below say how well the models work, not anything about real borrowers.

## What I did

First I cleaned the data and capped some extreme values. Then I made a few new columns from the existing ones: total late payments, the monthly debt payment (debt ratio times income), income per person in the household, and a flag for whether income was missing. A missing income can be a warning sign in itself, so I kept that information.

I split the data 80/20 into training and test sets, keeping the same share of defaulters in both. Then I trained three models: logistic regression, a decision tree and a random forest. Missing values are filled in with the median inside a pipeline, so nothing from the test set leaks into training. Because defaults are rare, I set the class weights to "balanced", otherwise the models mostly ignore the people I care about most.

## Results

These are on the 6,000 test applicants. Here "positive" means a person who defaulted.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.762 | 0.176 | 0.702 | 0.281 | 0.808 |
| Decision Tree | 0.768 | 0.169 | 0.639 | 0.268 | 0.781 |
| Random Forest | 0.829 | 0.214 | 0.586 | 0.313 | 0.802 |

The thing I found most interesting is that accuracy is almost useless here. A "model" that says nobody ever defaults is right 93.3% of the time, so the random forest's 82.9% accuracy actually looks worse than doing nothing. That's why I looked at recall and ROC-AUC instead.

Logistic regression ranked applicants best and caught about 70% of the people who defaulted. The catch is precision. Only 18% of the people it flags really do default, because defaults are so rare. I also tried different cut-offs: flagging anyone above 30% risk catches 88% of defaulters but with 11% precision, while a 70% cut-off drops recall to 46% and lifts precision to 29%. In a real bank, where you set that depends on what a bad loan costs compared to a turned-away customer.

The random forest says age, total late payments and credit utilization matter most. That just reflects how I built the fake data, so I wouldn't read anything into it.

## Running it

You need Python 3.9 or newer.

```
pip install -r requirements.txt
python credit_scoring.py
```

`credit_data.csv` is already in the repo. If you delete it, `python generate_data.py` makes it again. Charts, the results table and the saved model end up in the `outputs` folder. The charts open in pop-up windows, so close each one to let the script continue.

## What I'd do next

The obvious step is to try this on real data. If you have the Kaggle `cs-training.csv`, save it as `credit_data.csv` and run the script again. I wrote it to cope with that file's quirks, but I haven't been able to test it on the real thing. After that I'd try a gradient boosting model and look at how well the predicted probabilities match reality.
