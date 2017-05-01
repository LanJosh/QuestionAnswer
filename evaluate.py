"""
Functions for evaluating the performance of the model on the 
squad dataset

f1 and EM evaluations are from the squad evaluation scripts.
"""

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
  common = Couter(prediction_tokens) & Counter(answer_tokens)
  num_sum = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(answer_tokens)
  f1 = (2 * precision * recall) / (precision + recall) 
  return f1

def evaluate(predictions, answers):
  """
  Returns a tuple of (F1 score, EM score, sentence score)

  The sentence score is our evaluation method for determining the 
  effectiveness of finding the correct sentence within the context
  paragraph that may contain the answer. This metric is much softer
  than the F1 or EM score as it does not consider the difficulty in 
  finding the span within the sentence with the answer. The SQUAD 
  leaderboard and evaluation scripts only consider the F1 and EM score.
  """


