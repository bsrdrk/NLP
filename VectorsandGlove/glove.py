# -*- coding: utf-8 -*-
#İmport libraries .

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
#%% 
glove_input = 'C:/Users/Büşra/Desktop/NLP/Data/glove.6B.100d.txt'
word2vec_output = 'glove.6B.100d.word2vec'
glove2word2vec(glove_input,word2vec_output)

#%% 
model = KeyedVectors.load_word2vec_format(word2vec_output , binary = False)
#%%
print(model['nietzsche'])
print(model.most_similar('nietzsche'))
#finding queen
print(model.most_similar(positive=['woman' , 'king'] , negative=['man'] , topn = 1)) # it returns queen
print(model.most_similar(positive=['woman' , 'son'] , negative=['man'] , topn = 1))   #it returns daugther  
