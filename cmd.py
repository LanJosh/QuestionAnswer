"""
Script for running the QA system end to end

"""

from prepro import prepro
from evaluate import evaluate
import model
import baseline

data = prepro()
predictions = model.run()
results = evaluate(predictions, data['answers'])
print("Sentence score {} F1 score {}".format(results['sscore'], results['f1']))
baseline_predictions = baseline.run()
results = evaluate(baseline_predictions, data['answers'])
print("Baseline sentence score {}".format(data['sscore']))

