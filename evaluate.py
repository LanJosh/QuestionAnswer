"""
Functions for evaluating the performance of the model on the 
squad dataset

Modified the official squad dataset evaluation script
"""
import string
import re
from collections import Counter

def normalize_answer(s):
  """Lower text and remove punctuation, articles, and extra whitespace."""
  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, answer):
  prediction_tokens = normalize_answer(prediction).split()
  answer_tokens = normalize_answer(answer).split()
  common = Counter(prediction_tokens) & Counter(answer_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(answer_tokens)
  f1 = (2 * precision * recall) / (precision + recall) 
  return f1

def sentence_score(prediction, ground_truths):
  for ground_truth in ground_truths:
    if ground_truth in prediction:
      return 1
  return 0

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
  scores_for_ground_truths = []
  for ground_truth in ground_truths:
    score = metric_fn(prediction, ground_truth)
    scores_for_ground_truths.append(score)
  return max(scores_for_ground_truths)

def evaluate(predictions, answerss):
  """
  Returns a tuple of (F1 score, EM score, sentence score)

  The sentence score is our evaluation method for determining the 
  effectiveness of finding the correct sentence within the context
  paragraph that may contain the answer. This metric is much softer
  than the F1 or EM score as it does not consider the difficulty in 
  finding the span within the sentence with the answer. The SQUAD 
  leaderboard and evaluation scripts only consider the F1 and EM score.
  """
  f1 = sscore = total = 0 
  for prediction, answers in zip(predictions, answerss):
    total += 1
    f1 += metric_max_over_ground_truths(f1_score, prediction, answers)
    sscore += metric_max_over_ground_truths(sentence_score, prediction, answers)
  sscore = 100.0 * sscore / total
  f1 = 100.0 * f1 / total
  
  return {'sscore':sscore, 'f1':f1}

