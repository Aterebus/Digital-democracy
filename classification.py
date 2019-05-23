# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk
import sklearn 
from collections import Counter
import string
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import random

tag_map = {
        'CC':None, # coordin. conjunction (and, but, or)  
        'CD':wn.NOUN, # cardinal number (one, two)             
        'DT':None, # determiner (a, the)                    
        'EX':wn.ADV, # existential ‘there’ (there)           
        'FW':None, # foreign word (mea culpa)             
        'IN':wn.ADV, # preposition/sub-conj (of, in, by)   
        'JJ':[wn.ADJ, wn.ADJ_SAT], # adjective (yellow)                  
        'JJR':[wn.ADJ, wn.ADJ_SAT], # adj., comparative (bigger)          
        'JJS':[wn.ADJ, wn.ADJ_SAT], # adj., superlative (wildest)           
        'LS':None, # list item marker (1, 2, One)          
        'MD':None, # modal (can, should)                    
        'NN':wn.NOUN, # noun, sing. or mass (llama)          
        'NNS':wn.NOUN, # noun, plural (llamas)                  
        'NNP':wn.NOUN, # proper noun, sing. (IBM)              
        'NNPS':wn.NOUN, # proper noun, plural (Carolinas)
        'PDT':[wn.ADJ, wn.ADJ_SAT], # predeterminer (all, both)            
        'POS':None, # possessive ending (’s )               
        'PRP':None, # personal pronoun (I, you, he)     
        'PRP$':None, # possessive pronoun (your, one’s)    
        'RB':wn.ADV, # adverb (quickly, never)            
        'RBR':wn.ADV, # adverb, comparative (faster)        
        'RBS':wn.ADV, # adverb, superlative (fastest)     
        'RP':[wn.ADJ, wn.ADJ_SAT], # particle (up, off)
        'SYM':None, # symbol (+,%, &)
        'TO':None, # “to” (to)
        'UH':None, # interjection (ah, oops)
        'VB':wn.VERB, # verb base form (eat)
        'VBD':wn.VERB, # verb past tense (ate)
        'VBG':wn.VERB, # verb gerund (eating)
        'VBN':wn.VERB, # verb past participle (eaten)
        'VBP':wn.VERB, # verb non-3sg pres (eat)
        'VBZ':wn.VERB, # verb 3sg pres (eats)
        'WDT':None, # wh-determiner (which, that)
        'WP':None, # wh-pronoun (what, who)
        'WP$':None, # possessive (wh- whose)
        'WRB':None, # wh-adverb (how, where)
        '$':None, #  dollar sign ($)
        '#':None, # pound sign (#)
        '“':None, # left quote (‘ or “)
        '”':None, # right quote (’ or ”)
        '(':None, # left parenthesis ([, (, {, <)
        ')':None, # right parenthesis (], ), }, >)
        ',':None, # comma (,)
        '.':None, # sentence-final punc (. ! ?)
        ':':None # mid-sentence punc (: ; ... – -)
    }


def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    text1=''
    for w in text:
       text1=text1+w.lower()
    text2=text1
        
    text2=nltk.word_tokenize(text2)
    posit=nltk.pos_tag(text2)
    list1=[]
    for item, POS in posit: 
        list1.append(item)        
    return list1


def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    df['text']=df['text'].apply(process)
    return df;
    
def get_rare_words(n, processed_tweets):
    y=Counter()
    Twitter_text=processed_tweets['text']
    for r  in Twitter_text:
        y=y+Counter(r)
    rare_words=[]
    for w in y:
        if (y[w]<n):
           rare_words.append(w)
    rare_words.sort()
    return rare_words
    
def mirror_tokenizer(x):
    return x    
    
def create_features(processed_tweets, rare_words):
    stop_words=nltk.corpus.stopwords.words('english')
    Cleared_processed_tweets=processed_tweets
    scop=[]
    for r in Cleared_processed_tweets['text']:
        r= [w for w in r if not w in stop_words]
        r= [w for w in r if not w in rare_words]
        scop=scop+[r]
    tfidf = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=mirror_tokenizer, lowercase=False)
    tfidf=tfidf.fit(scop)
    X=tfidf.transform(scop)
    return (tfidf, X)
    
def create_labels(processed_tweets):
    democrats=['HillaryClinton','TheDemocrats','timkaine']
    labels=[]
    for users in processed_tweets['screen_name']:
          if users in democrats:
            labels.append(0)
          else:
            labels.append(1)
    labels=np.asarray(labels)
    return labels
    
def learn_classifier(X_train, y_train, kernel):
    init=sklearn.svm.classes.SVC(kernel=kernel)
    classifier=init.fit(X_train, y_train)
    return classifier


def evaluate_classifier(classifier, X_validation, y_validation):
    ev=classifier.predict(X_validation)
    accuracy=1-sum(abs(y_validation-ev)/len(ev))
    return accuracy
    
def classify_tweets(tfidf, classifier, unlabeled_tweets):
    scop=[]
    unlabeled=process_all(unlabeled_tweets)
    for r in unlabeled['text']:
        scop=scop+[r]
    X_unlabel=tfidf.transform(scop)
    predicted_label=classifier.predict(X_unlabel)
    return predicted_label


for rare in range(1,2):
   tweets = pd.read_csv("tweets_train.csv", na_filter=False)
   processed_tweets = process_all(tweets)
   rare_words = get_rare_words(rare, processed_tweets)
   (tfidf, X) = create_features(processed_tweets, rare_words)
   y = create_labels(processed_tweets)
   total_number=len(processed_tweets['text'])
   train_number=round(0.8*total_number)
   random_vals=random.sample(range(total_number+1), train_number)
   X_train=X[[i for i in range(0,total_number) if i in random_vals]]
   X_valid=X[[i for i in range(0,total_number) if i not in random_vals]]
   Y_train=y[[i for i in range(0,total_number) if i in random_vals]]
   Y_valid=y[[i for i in range(0,total_number) if i not in random_vals]]
   classifier = learn_classifier(X_train, Y_train, 'linear')
   accuracy = evaluate_classifier(classifier, X_valid, Y_valid)
   print(accuracy)

unlabeled_tweets = pd.read_csv("coffee.csv", na_filter=False)
y_pred = classify_tweets(tfidf, classifier, unlabeled_tweets)

dem=0
rep=0

for i in range(len(y_pred)):
    if (y_pred[i]==0):
        dem=dem+1
    if (y_pred[i]==1):
        rep=rep+1
        
print('Democrats ',dem)
print('Republicans ',rep)


