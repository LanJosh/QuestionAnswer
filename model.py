"""
Model for extracting answer sentence from the context paragraph.

"""


from nltk.tokenize.moses import MosesDetokenizer
from tqdm import tqdm
import json
import numpy as np

def cosine_similarity(x, y):
  """
  Compute the cosine similarity between two vectors x and y
  """
  return (x.dot(y)) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))

def sent_embed(x, word2vec):
  """
  Combine the word vectors of a sentence to get a single vector
  representation of the entire sentence
  """
  sent_vec = np.zeros((300,))
  for word in x:
    if word in word2vec:
      sent_vec = np.add(sent_vec, np.array(word2vec[word]))
  return sent_vec 

def run():
  detokenizer = MosesDetokenizer() 
  with open('data.json') as f:
    data = json.load(f)
  word2vec = data['word2vec']
  contexts = data['contexts']
  questions = data['questions']
  predictions = []
  for c,qs in tqdm(zip(contexts, questions), total=len(contexts)):

    # Get vector embedding of context
    ce = []
    for sent in c:
      ct = sent_embed(sent,word2vec)
      ce.append(ct)

    # Get vector embedding of sentence
    # Find the most similar sentence in the context
    for q in qs:
      qe = sent_embed(q,word2vec)
      sims = [cosine_similarity(qe, cs) for cs in ce]
      max_sim = max(sims)
      idx = sims.index(max_sim)
      predictions.append(detokenizer.detokenize(c[idx], return_str=True))
  return predictions 

