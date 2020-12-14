# -*- coding: utf-8 -*-
#import libraries
import numpy as np 
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#%% read data and tokenization
f = open('C:/Users/Büşra/Desktop/NLP/Data/hurriyet.txt' , 'r' , encoding = 'utf8')
text = f.read() 
text_list = text.split('\n')
corpus = []
for sentence in text_list:
    corpus.append(sentence.split())
#%% create model with skip gram
model = Word2Vec(corpus , size = 100 , window=5 , min_count=5 , sg = 1)

#%% Check
print(model.wv['ankara'])
print(model.wv.most_similar('pazartesi'))

# %% closest plot 
def closest_words(model,word):
    word_vectors = np.empty((0,100))
    word_labels = [word]
    
    close_words = model.wv.most_similar(word)
    
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]) , axis = 0)
    for w, score in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors,np.array([model.wv[w]]) , axis = 0)
    tsne = TSNE(random_state=0)
    Y= tsne.fit_transform(word_vectors)
    
    x_coord = Y[:, 0]
    y_coord = Y[:, 1]
    
    plt.scatter(x_coord , y_coord)
    for label, x,y in zip(word_labels,x_coord,y_coord):
        plt.annotate(label,xy=(x,y) , xytext=(5,-2), textcoords='offset points')
    plt.show()
#%%    
closest_words(model,'pazartesi')