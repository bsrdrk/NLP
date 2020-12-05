# -*- coding: utf-8 -*-
import nltk 
import io
import zemberek
from os.path import join
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM , java
from zemberek import (TurkishMorphology)
import jpype as jp
import string
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter
# %% data 
filename = 'gazete.txt' 
with io.open(filename, 'r', encoding='utf8') as f:
    text = f.read()
text = text.lower()    
# %% tokenize 
tokens = nltk.word_tokenize(text , language='turkish') 
tokens = [''.join(c for c in s if c not in string.punctuation) for s in tokens] 
sentence_tokens = nltk.sent_tokenize(text , language='turkish') 
sentence_tokens = [''.join(c for c in s if c not in string.punctuation) for s in sentence_tokens]
# %% lemma
lemmas_list = []
lemma_sentence = []
lemma_dict = {}
if __name__ == '__main__':
    ZEMBEREK_PATH: str = join('zemberek-full.jar')

    startJVM( # dogru olan
        getDefaultJVMPath(), 
        '-ea', 
        '-Djava.class.path=%s' % (ZEMBEREK_PATH)
        )  

    morphology = JClass('zemberek.morphology.TurkishMorphology').createWithDefaults()
    WordAnalysis = JClass('zemberek.morphology.analysis.WordAnalysis')
    lemmas = {}
    for sentence in sentence_tokens:
        if sentence != '':
            analysis: java.util.ArrayList = (
                morphology.analyzeAndDisambiguate(sentence).bestAnalysis()
            )
        
            lemma = []
            
            for i, analysis in enumerate(analysis, start=1):
                lemma.append(f'{str(analysis.getLemmas()[0])}')
                lemma_dict[f'{str(analysis.getLemmas()[0])}'] = pos_dict.get(f'{str(analysis.getLemmas()[0])}' , 0.0) + 1.0 
                lemmas_list.append(f'{str(analysis.getLemmas()[0])}')
            #print(f'\nFull sentence with POS tags: {" ".join(lemma)}')
            lemma_sentence.append(f'{" ".join(lemma)}')

# %% ngram
bigram = ngrams(lemmas_list, 2)
bigram_freq = Counter(bigram)

trigram = ngrams(lemmas_list, 3)
trigram_freq = Counter(trigram)

bigram_freq_list = list(bigram_freq.values())
bigram_list = list(bigram_freq)
trigram_freq_list = list(trigram_freq.values())
trigram_list = list(trigram_freq)
trigram_at_least_5 = []   
bigram_at_least_5 = []  
trigram_at_least_5dict = {}   
bigram_at_least_5dict = {}  
for i in range(0,len(bigram_list)):

    if ( bigram_freq_list[i] >= 5):
        bigram_at_least_5dict[bigram_list[i]] = bigram_freq_list[i]
        bigram_at_least_5.append(bigram_list[i])
        print(bigram_list[i] , '    ' , bigram_freq_list[i] , '   ')
    if ( trigram_freq_list[i] >= 5 ):
        trigram_at_least_5dict[trigram_list[i]] = trigram_freq_list[i]
        trigram_at_least_5.append(trigram_list[i])
        print(trigram_list[i] , '    ' , trigram_freq_list[i] , '   ')
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
stopWords = set(stopwords.words('turkish'))
documents = lemma_sentence
vectorizer = TfidfVectorizer(stop_words=stopWords, 
                             use_idf=True, 
                             ngram_range=(1,3))
X = vectorizer.fit_transform(documents)

# %%
lsa = TruncatedSVD(n_components=3565 , n_iter=10)
lsa.fit(X)

# %% 
terms= vectorizer.get_feature_names()
for i , comp in enumerate(lsa.components_):
    termsInComp = zip(terms, comp)
    sortedterms = sorted(termsInComp, key = lambda x: x[1], reverse= True)[0:10]
    if i==10:
        break
    print( i )
    for term in sortedterms[0:10]:
        print(term[0])
    print(' ')
   
