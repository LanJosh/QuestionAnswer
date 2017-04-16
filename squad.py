"""
Utility functions for loading and preprocessing data.
Fork of https://github.com/allenai/bi-att-flow/

"""

import os
import json
import nltk
import re
from collections import Counter

def word_tokenize(tokens):
  return [token.replace("''", '"').replace("``",'"') for token in nltk.word_tokenize(tokens)]

def get_word_idx(context, wordss, idx):
  spanss = get_2d_spans(context, wordss)
  return spanss[idx[0]][idx[1]][0]

def get_2d_spans(text, tokenss):
  """
  Get the span of each token in the context paragraph

  """
  spanss = []
  cur_idx = 0
  for tokens in tokenss:
    spans = []
    for token in tokens:
      if text.find(token, cur_idx) < 0:
        print(tokens)
        print("{} {} {}".format(token, cur_idx, text))
        raise Exception()
      cur_idx = text.find(token, cur_idx)
      spans.append((cur_idx, cur_idx + len(token)))
      cur_idx += len(token)
    spanss.append(spans)
  return spanss

def get_word_span(context, wordss, start, stop):
  """
  Find the (sentence, word) index into wordss of the words within the 
  start/stop range.
  ```context``` is the paragraph 
  ```wordss``` is the paragraph tokenized into sentences, tokenized into
  words
  ```start``` and ```stop``` are both character level indices into context
  """
  spanss = get_2d_spans(context, wordss)
  idxs = []
  for sent_idx, spans in enumerate(spanss):
    for word_idx, span in enumerate(spans):
      if not (stop <= span[0] or start >= span[1]):
        idxs.append((sent_idx, word_idx))

  assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
  return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)

def get_word2vec(words, dim=50):
  """
  dim is the number of dimensions for the glove pretrained embeddings. 
  """
  source_dir = os.path.dirname(os.path.realpath(__file__))
  data_dir = os.path.join(source_dir, 'datasets')
  glove = os.path.join(data_dir, "glove.6B.{}d.txt".format(dim))
  with open(glove, 'r') as f:
    for line in f:
      ary = line.lstrip().rsplit().split(" ")
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

def prepro_squad(data_dir=None):
  """
  ```q``` is the word tokenized questions
  ```cq``` is the character level ```q```
  ```y``` is a tuple of the sent,word index of the first and last words
  in the answer text
  ```rx``` is the (article index, paragraph index)
  ```rcx``` is the (article index, paragraph index)
  ```ids``` is the question ids
  ```idxs``` is a count id for each question

  """

  q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
  cy = []
  x, cx = [], []    # context and context characters
  answerss = []     # answer texts 
  p = []            # Unprocessed context paragraphs
  words = Counter() # multiset of all the words in the data

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

  for ai, article in enumerate(train_data['data']):
    # Context per article, i.e. xp[0] is all the context paragraphs 
    # for Super Bowl 50
    xp, cxp = [], [] 
    pp = []
    x.append(xp)
    cx.append(cxp)
    p.append(pp)
    for pi, paragraph in enumerate(article['paragraphs']):
      context = paragraph["context"]
      context.replace("''", '"')
      context.replace("``", '"')
      # Context per paragraph
      xi = list(map(word_tokenize, nltk.sent_tokenize(context)))
      xi = [process_tokens(tokens) for tokens in xi] 
      cxi = [[list(xijk) for xijk in xij] for xij in xi]
      xp.append(xi)
      cxp.append(cxi)
      pp.append(context)

      for sent in xi:
        for word in sent:
          words[word] += len(paragraph['qas'])

      rxi = [ai, pi] 
      assert len(x) - 1 == ai
      assert len(x[ai]) - 1 == pi
      for qa in paragraph['qas']:
        qi = word_tokenize(qa['question']) 
        cqi = [list(qij) for qij in qi]

        # The (sentence, word) index of the answer
        yi = [] 
        # The character index of the answer 
        cyi = []
        answers = []
        for answer in qa['answers']:
          answer_text = answer['text']
          answers.append(answer_text)
          answer_start = answer['answer_start']
          answer_stop = answer_start + len(answer_text)
          yi0, yi1 = get_word_span(context, xi, answer_start, answer_stop)
          assert len(xi[yi0[0]]) > yi0[1]
          assert len(xi[yi1[0]]) >= yi1[1]
          answer_start_word = xi[yi0[0]][yi0[1]]
          answer_end_word = xi[yi1[0]][yi1[1]-1]
          i0 = get_word_idx(context, xi, yi0)
          i1 = get_word_idx(context, xi, (yi1[0], yi1[1]-1))
          cyi0 = answer_start - i0
          cyi1 = answer_stop - i1 - 1
          assert answer_text[0] == answer_start_word[cyi0], (answer_text, w0, cyi0)
          assert answer_text[-1] == answer_end_word[cyi1]

          yi.append([yi0, yi1])
          cyi.append([cyi0, cyi1])

        q.append(qi)
        cq.append(cqi)
        y.append(yi)
        cy.append(cyi)
        rx.append(rxi)
        rcx.append(rxi)
        ids.append(qa['id'])
        idxs.append(len(idxs))
        answerss.append(answers)

  word2vec = get_word2vec(word_counter)

  return word2vec, x, cx, q, cq, answerss, y 


