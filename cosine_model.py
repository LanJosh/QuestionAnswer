"""
Find sentence to query similarity using word2vec and the cosine distance.
The most similar sentence in the context paragraph should have the 
answer within it.

"""

import os
import json
import nltk
import re
from collections import Counter
from tqdm import tqdm

def word_tokenize(tokens):
  return [token.replace("''", '"').replace("``",'"') for token in nltk.word_tokenize(tokens)]

def get_word2vec(words):
  """
  dim is the number of dimensions for the glove pretrained embeddings. 
  """
  source_dir = os.path.dirname(os.path.realpath(__file__))
  data_dir = os.path.join(source_dir, 'datasets')
  glove = os.path.join(data_dir, "glove.42B.300d.txt")
  word2vec = {}
  total = 1917494
  with open(glove, 'r') as f:
    for line in tqdm(f, total=total):
      ary = line.lstrip().rstrip().split(" ")
      word = ary[0]
      vector = list(map(float,ary[1:]))
      if word in words:
        word2vec[word] = vector
      elif word.capitalize() in words:
        word2vec[word.capitalize()] = vector
      elif word.lower() in words:
        word2vec[word.lower()] = vector
      elif word.upper() in words:
        word2vec[word.upper()] = vector

  print("{}/{} words in data have pretrained embeddings".format(len(word2vec), len(words)))

  return word2vec

def process_tokens(temp_tokens):
  tokens = [] 
  for token in temp_tokens:
    flag = False
    l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
    tokens.extend(re.split("([{}])".format("".join(l)), token))
  return tokens

def prepro(data_dir=None):
  """
  Get each of the words in the dataset to create a word2vec lookup d
  dictionary.
  Perform some preprocessing on tokens for model training. 

  """

  if data_dir == None:
    source_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(source_dir, 'datasets')
    train_path = os.path.join(data_dir, 'train-v1.1.json')
    dev_path = os.path.join(data_dir, 'dev-v1.1.json')
    with open(train_path) as f:
      train_data = json.load(f)
    with open(dev_path) as f:
      dev_data = json.load(f) 
    train_data["data"].extend(dev_data["data"]) 
  else:
    with open(data_dir) as f:
      train_data = json.load(f)

  contexts, questions, answers = [], [], [] 
  words = set() 
  for article in tqdm(train_data['data']):
    for paragraph in article['paragraphs']:
      context = paragraph["context"]
      context.replace("''", '"')
      context.replace("``", '"')
      # Context per paragraph
      xi = list(map(word_tokenize, nltk.sent_tokenize(context)))
      xi = [process_tokens(tokens) for tokens in xi] 
      contexts.append(xi)

      for sent in xi:
        for word in sent:
          words.add(word)

      for qa in paragraph['qas']:
        qi = word_tokenize(qa['question']) 
        for word in qi:
          words.add(word)
        questions.append(qi)

        for answer in qa['answers']:
          answer_text = answer['text']
          answers.append(answer_text)

  word2vec = get_word2vec(words)

  data = { 'word2vec' : word2vec, 'contexts' : contexts, 'questions' : questions, 'answers' : answers }

  return data

