"""
A naive baseline implementation for handling question
answer pairs. The question and passage are transformed
using pretrained word vectors. Each sentence in the question
and passage are scored. The sentence in the passage
with the closest score to the question is taken to 
have the answer.

"""


