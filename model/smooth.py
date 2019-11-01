# proportions -  proportion [0-1] of each activity in the dataset
# conf - confusion matrix, normalized so the sum of each row is 1
# trans - transition matrix (probability of transitions from one activity to another for each pair). Normalized in the same manner.
# pred - Current predictions to be smoothed
# pred_smoothed - Smoothed predictions

from hmmlearn import hmm
import pandas as pd

activities = ["Still", "Walking", "Run", "Bike", "Car", "Bus", "Train", "Subway"]

model = hmm.MultinomialHMM(n_components=len(activities))
model.startprob_ = proportions
model.transmat_ = trans
model.emissionprob_ = conf
# The "lambda function" in following lines simply transform the activities from string to integer and back
pred_smoothed = model.predict(np.array(pred.apply(lambda x: list(activities).index(x))).reshape([len(pred), 1]))
pred_smoothed = pd.pred_smoothed(pred).apply(lambda x: activities[x])
