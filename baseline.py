"""
Model for extracting answer sentence from the context paragraph.

"""

import random
from nltk.tokenize.moses import MosesDetokenizer
from tqdm import tqdm
import json
import numpy as np

def run():
  detokenizer = MosesDetokenizer() 
  with open('data.json') as f:
    data = json.load(f)
  contexts = data['contexts']
  questions = data['questions']
  predictions = []
  for c,qs in tqdm(zip(contexts, questions), total=len(contexts)):
    if len(c) == 1:
      continue
    # Get vector embedding of sentence
    # Find the most similar sentence in the context
    for q in qs:
      predictions.append(detokenizer.detokenize(random.choice(c), return_str=True))
  return predictions 

