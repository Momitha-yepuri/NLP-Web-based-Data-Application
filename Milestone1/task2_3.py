#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Momitha
# #### Student ID: s3856512
# 
# 
# Date: 3/11
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# In this task, we generated 7 different types of feature representations for the collection of job advertisements. Only considering the description of the job advertisement. The feature representation that we generate include:
# - `Bag-of-words` model
# -` FastText` language model (Word2Vec pretrained model)
# - `GloVe` (Word2Vec pretrained model)
# 
# For the above models we computed the weighted (i.e., TF-IDF weighted) and unweighted vector representation for each job advertisement description.

# ## Importing libraries 

# In[1]:


# Code to import libraries as you need in this assessment, e.g.,
import nltk
import pandas as pd
import numpy as np
from sklearn.datasets import load_files  
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import *
from itertools import chain
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk.data
import re
import os


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# ### 2.1 The Dataset

# In the previous task1, we have demonstrated the basic steps of text pre-processing with the description of the job advertisements. 
# We have also saved all the tokenized description, the vocabulary and the generated data features in .txt files. 
# In this task, we are going to bulid maching leaning models for document classification, using our generated data features. 
# 
# Same as in the previous activity, the document collection that we are going to use is the `job_ads.txt`. 
# To be brief, the dataset:
# - Category Labels: 8 (Accounting_Finance, Engineering, Healthcare_Nursing, Hospitality_Catering, IT, PR_Advertising_Marketing, Sales and Teaching.)
# 
# - `job_ads.txt` stores the pre-processed description,ID of the job ads and and the unprocessed titles, WebIndexed and categories.
# - `Vocab.txt` stores the vocabulary (as well as the index of each word).
# 
# #### Importing data from task 1 using Regex
# Iterating through the all the .txt files in all 8 folders and storing each field in a list.

# In[2]:


# Code to perform the task...
# Reference: https://www.programcreek.com/python/example/96665/re.fullmatch

txt_file = 'job_ads.txt'

job_id=[]
job_category=[]
job_web_index=[]
job_title=[]
job_description=[]

with open(txt_file) as f: 
    fileread = f.read().splitlines() 
for i in fileread:
    if re.fullmatch(r'ID: (\d{5})',i):
        ID=re.fullmatch(r'ID: (\d{5})',i)
        job_id.append(ID.group(1))

    if re.fullmatch(r'Category: (.+)',i):
        Cat=re.fullmatch(r'Category: (.+)',i)
        job_category.append(Cat.group(1))

    if re.fullmatch(r'Webindex: (.+)',i):
        WI=re.fullmatch(r'Webindex: (.+)',i)
        job_web_index.append(WI.group(1))

    if re.fullmatch(r'Title: (.+)',i):
        T=re.fullmatch(r'Title: (.+)',i)
        job_title.append(T.group(1))

    if re.fullmatch(r'Description: (.+)',i):
        D=re.fullmatch(r'Description: (.+)',i)
        job_description.append(D.group(1))
        


# In[3]:


# Converting the vocab file into a list from 
# task 1 and splitting by a colon to just the 
with open("vocab.txt", "r") as x:
    lines = x.readlines()
    vocab = []
    for l in lines:
        as_list = l.split(":")
        vocab.append(as_list[0])
    vocab[0]


# In[4]:


#checking the length
len(vocab)


# In[5]:


#checking the length
len(job_description)


# In[6]:


#checking the datatype
type(vocab),type(job_description)


# In[7]:


# test index to check if fields are right - used throughout the notebook
test_ind = 2


# In[8]:


# checking ID, webIndex, category and title of a specific job ad
job_id[test_ind], job_web_index[test_ind],job_category[test_ind], job_title[test_ind], 


# In[9]:


# checking descriptions of a specific job ad
job_description[test_ind]


# In[10]:


tk_description=[description.split(' ') for description in job_description]


# ### 2.2 Building Vector Representation
# After text pre-processing has been completed for description in job_ad, each individual document needs to be transformed into a numeric representation.
# In the set of job adverstisements and a given pre-defined list of words(vocab). We are computing a count vector representation for each description. 
# * an integer count, each entry is `word:count`, telling how many times a word appear in a document.
# 
# ### 2.2.1 Bag of words - Generating Count Vector 
# We will demonstrate the usage of the following two classes:
# CountVectorizer: It converts a collection of text documents to a matrix of token counts.

# In[11]:


cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer


# In[12]:


count_features = cVectorizer.fit_transform([description for description in job_description]) # generate the count vector representation for all descriptions
print(count_features.shape)


# In[13]:


job_description[2]


# In[14]:


# test_ind = 2


# #### Check Count Vector Representation
# We are printing out the words that appear in a description, by out running example `test_ind`. 

# In[15]:


def validator(data_features, vocab, a_ind,job_description):
    print("Job Description:",job_description[a_ind])
    print("--------------------------------------------\n")
    print("Vector representation:\n") # printing the vector representation as format 'word:value' (
                                      # an integer for count vector; 
    for word, value in zip(vocab, data_features.toarray()[test_ind]): 
        if value > 0:
            print(word+":"+str(value), end =' ')


# In the validator function above, we used a`zip` function that takes iterable items and merges them into a single tuple. The resultant value is a zip object that stores pairs of iterables.
# 
# The vocab directly corresponds to the data_features.The `toarray()` method converts the efficient representation of a sparse matrix that sklearn uses to an ordinary readable dense ndarray representation.
# 

# In[16]:


validator(count_features,vocab,test_ind,job_description)


# In[17]:


FreqDist(tk_description[test_ind])


# Also manually checked some of the frequencies values and compare with the original .txt file to see whether I've done that properly.

# ### 2.2.2 Saving Output Count Vector
# 
# This file stores the sparse count vector representation of job advertisement descriptions in the following format. Each line starts with a ‘#’ key followed by the webindex of the job advertisement, and a comma ‘,’. The rest of the line is the sparse representation of the corresponding description in the form of `word_integer_index:word_freq` separated by comma.

# In[18]:


filename = "count_vectors.txt" # File name
job_total = count_features.shape[0] # Number of job descriptions
out_file = open(filename, 'w') 
for i in range(0, job_total): # loop through every job ad
    out_file.write('#'+str(job_web_index[i]).strip('\n'))
    for vocab_ind in count_features[i].nonzero()[1]: # for each word index that has non-zero entry in the count_features
        value = count_features[i][0,vocab_ind] 
        out_file.write(",{}:{}".format(vocab_ind,value)) 
    out_file.write('\n') # start a new line after each description
out_file.close() 


# ### 2.3 Converting into a dataframe

# In[19]:


# converting into a pd dataframe for see TDIDF weighted and 
# unweighted score of each model.
df=pd.DataFrame()
df['jobId']=job_id
df['jobWebindex']=job_web_index
df['jobCategory']=job_category
df['jobTitle']=job_title
df['jobDescription']=job_description
df['tkdescription']=tk_description
df.head()


# In[20]:


df.shape #no. of rows & columns


# The `gen_vocIndex` function reads the the vocabulary file, and create an `w_index:word` dictionary for us to create `job_tVectors.txt`

# In[21]:


def gen_vocIndex(voc):
    return {int(i):voc[i] for i in range(0,len(voc))}


# In[22]:


vocab_dict = gen_vocIndex(vocab)
vocab_dict


# ### 2.4 Generating TF-IDF Vectors

# We will generate the TF-IDF Vector to represent each of the document.
# 
# Similar to the use of `CountVector`, we first initialise a `TfidfVectorizer` object by only specifying the value of "analyzer" and the vocabulary, and then convert the job descriptions into a list of strings, each of which corresponds
# to a job description.

# In[23]:


from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform([' '.join(description) for description in tk_description]) # generate the tfidf vector representation for all descriptions
tfidf_features.shape


# In[24]:


validator(tfidf_features,vocab,test_ind,job_description)


# Printing out the weighted vector for the example document.

# ### 2.4.1 Saving the Vector Representation

# Here, we are saving the `tdidf` vector representation. As there are only a limited number of words appear in a document, we are retrieving the index of the non-zero entry of the data features by calling the `nonzero()` function. 
# This function checks and return the indices of the elements that are non-zero. These indices are returned as a tuple of arrays, one for each dimension of the matrix, containing the indices of the non-zero elements in that dimension.
# 
# The `job_tVector.txt` is saving the word index of each word that appears in the description in the form of `word_integer_index:word_weight `separated by colon.

# In[25]:


filename = "job_tVector.txt"
num = tfidf_features.shape[0] # the number of document
out_file = open(filename, 'w') # careates a txt file and open to save the vector representation
for a_ind in range(0, num): # loops through each descriptiona by index
    for f_ind in tfidf_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
        value = tfidf_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
        out_file.write("{}:{} ".format(f_ind,value)) # write the entry to the file in the format of word_index:value
    out_file.write('\n') # start a new line after each description
out_file.close() # close the file


# The doc_wordweights function takes the tfidf document vector file, as well as the `w_index:word` vocab dictionary, creates the mapping between `w_index` and the actual word, and creates a dictionary of `word:weight` or each unique word appear in the document.

# In[26]:


def doc_wordweights(fName_tVectors, vocab_dict):
    tfidf_weights = [] # a list to store the  word:weight dictionaries of documents

    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector
        tv = tv.strip()
        weights = tv.split(' ') # list of 'word_index:weight' entries
        weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
        wordweight = {vocab_dict[int(w[0])]:w[1] for w in weights} # construct the weight dictionary, where each entry is 'word:weight'
        tfidf_weights.append(wordweight) 
    return tfidf_weights

fName_tVectors = 'job_tVector.txt' #give t vector....
tfidf_weights = doc_wordweights(fName_tVectors, vocab_dict)


# In[27]:


# taking a look at the tfidf word weights dictionary of the first document
tfidf_weights[0]


# ### 2.5 FastText Model

# Using the fastText library for training word-embedding models, and performing similarity operations & vector lookups analogous to Word2Vec. 
# 
# In the following block of code, we import the `FastText` model form Gensim library, then:
# 1. We set the path to the corpus file. Similar as above, we use the `descriptions.txt` as the training corpus;
# 2. Initialise the `FastText` model, similar as before, we use 200 dimention vectors;
# 3. Then we build the vocabulary from the copurs;
# 4. Finally, we train the fasttext model based on the corpus.

# In[28]:


import gensim.downloader as api


# In[29]:


out_file = open("descriptions.txt", 'w') # creates a txt file named 'descriptions.txt', open in write mode
for desc in tk_description:
    out_file.write(' '.join(desc) + '\n') # join the tokens in a description with space, and write the obtained string to the txt document
out_file.close() # close the file


# In[30]:


from gensim.models.fasttext import FastText

#corpus file name
corpus_file = 'descriptions.txt'

# Initialising FastText model with vectorsize of 200
descFT = FastText(vector_size=200) 

# providing the corpus
descFT.build_vocab(corpus_file=corpus_file)

# training the model
descFT.train(
    corpus_file=corpus_file, epochs=descFT.epochs,
    total_examples=descFT.corpus_count, total_words=descFT.corpus_total_words,
)

print(descFT)


# In[31]:


descriptionFT = descFT
print(descriptionFT)
descriptionFT_wv= descriptionFT.wv


# The `gen_docVecs` function was taken from actvity 6.
# This function has been modified for better performance and faster runtime. It takes the word embeddings dictionary, the tokenized text of descriptions, and the tfidf weights (list of word:weight dictionaries, one for each description) as arguments, and generates the document embeddings:
#  1. creates an empty list `docs_vectors` to store the document embeddings of descriptions
#  2. it loops through every tokenized description:
#     - creates an empty list `temp` to store the word vectors and word weights in every description
#     - for each word that exists in the word embeddings dictionary/keyedvectors, 
#         - if the argument `tfidf` weights are empty `[]`, it sets the weight of the word as 1
#         - otherwise, retrieve the weight of the word from the corresponding word:weight dictionary of the description from  `tfidf`
#     - row bind the weighted word embedding to `temp`
#     - takes the sum of each column to create the document vector, i.e., the embedding of a description
#     - append the created document vector to the list of document vectors
#     - convert the `doc_vectors` back to a pd dataframe to see the model scores.
# 

# In[32]:


def gen_docVecs(wv,tk_txts,tfidf = []): 
    # Generating Vector Representation
    # if tfidf parameter is empty==> unweighted
    # if tfidf paramter is not empty=>weighted

    docs_vectors = [] #Initialising Empty dataframe
    for i in range(0,len(tk_txts)):
        tokens = list(set(tk_txts[i])) # Distinct words in the document

        temp = []  #empty list to store and sum the word embeddings of a document
        for w_ind in range(0, len(tokens)): # Iterating over distinct words in a document
            try:
                word = tokens[w_ind]
                word_vec = wv[word] # if word is present in pretrained model's dimension we execute further else 
                                    #exception occurs and iteration skips

                if tfidf != []: # if tfidf parameter is non empty it is for weighted and we assign the weights
                    word_weight = float(tfidf[i][word])
                else:
                    word_weight = 1 # if the parameter is empty it is unweighted and we assign weight=1
                temp.append(word_vec*word_weight) #if word is present multiply the weights and append in the dataframe
            except:
                pass
        doc_vector = np.sum(temp,axis=0) # When the iteration is over for a whole document 
                                        #we add all the word embedding for the document and store it.
        docs_vectors.extend([doc_vector])
    docs_vectors=pd.DataFrame(docs_vectors) # output as a dataframe
    return docs_vectors


# ### 2.5.1 Unweighted and Weighted FastText

# In[33]:


#Unweighted
descriptionFT_dvs = gen_docVecs(descriptionFT_wv,df['tkdescription'])


# In[34]:


descriptionFT_dvs.head()


# In[35]:


#weighted
Weighted_descriptionFT_dvs = gen_docVecs(descriptionFT_wv,df['tkdescription'],tfidf_weights)


# In[36]:


Weighted_descriptionFT_dvs.head()


# ### 2.5.2 FastText - Check weighted and unweighted 

# In[37]:


descriptionFT_dvs.isna().any().sum() 


# In[38]:


descriptionFT_dvs.shape


# In[39]:


Weighted_descriptionFT_dvs.isna().any().sum() # check whether there is 


# In[40]:


Weighted_descriptionFT_dvs.shape


# In[41]:


FT = gen_docVecs(descriptionFT_wv,df['tkdescription'][0:100])


# ### 2.5.3 FastText Visualisations

# The following function `plotTSNE` is used to plot Fast Text, GloVe and Googlenews visulisations.
# It takes the following arugments:
# 
# - labels, the lable/category of each job advertisement.
# - features, a numpy array of document embeddings, each for a job ad.
# and projects the feature/document embedding vectors in a 2 dimension space and plot them out. It does the following:
# 
# 1. get the set of classes, called categories (8 categories)
# 2. sample 5% of the data/document embeddings randomly, and record the indices selected
# 3. project the selected document embeddings in 2 dimensional space using tSNE, each document embedding now corresponds to a 2 dimensional vector in projected_features
# 4. plot them out as scatter plot and highlight different categories in different color

# In[42]:



def plotTSNE(labels,features): # features as a numpy array, each element of the array is the document embedding of a description
    job_categories = sorted(labels.unique())
    # Sampling a subset of our dataset because t-SNE is computationally expensive
    SAMPLE_SIZE = int(len(features) * 0.05)
    np.random.seed(0)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = TSNE(n_components=2, random_state=0).fit_transform(features[indices])
    colors = ['pink', 'orange', 'midnightblue', 'darkgrey', 'green','black','blue','yellow']
    for i in range(0,len(job_categories)):
        points = projected_features[(labels[indices] == job_categories[i])]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=job_categories[i])
    plt.title("Feature vector for each category, projected on 2 dimensions.",
              fontdict=dict(fontsize=15))
    plt.legend()
    plt.show()
    


# In[43]:


#UnWeighted
features = descriptionFT_dvs.to_numpy()
plotTSNE(df['jobCategory'],features)


# In[ ]:





# In[44]:


#Weighted
features = Weighted_descriptionFT_dvs.to_numpy()
plotTSNE(df['jobCategory'],features)


# In[ ]:





# ## 2.6 GloVe

# In[45]:


# from utils import loadGloVe

fPath = "glove/glove.6B.200d.txt"
# preTGloVe_wv = loadGloVe(fPath)

new = {} # initialise an empty dictionary
with open(fPath, encoding="utf-8") as f: # open the txt file containing the word embedding vectors
    for line in f:
        word, coefs = line.split(maxsplit=1) # The maxsplit defines the maximum number of splits. 
                                             # in the above example, it will give:
                                             # ['population','0.035182 1.4248 0.9758 0.1313 -0.66877 0.8539 -0.11525 ......']
        coefs = np.fromstring(coefs, "f", sep=" ") # construct an numpy array from the string 'coefs', 
                                                   # e.g., '0.035182 1.4248 0.9758 0.1313 -0.66877 0.8539 -0.11525 ......'
        new[word] = coefs # create the word and embedding vector mapping

print("Found %s word vectors." % len(new))


# ### 2.6.1 GloVe Unweighted and Weighted 

# In[46]:


# Unweighted
GloVe_desc_dvs = gen_docVecs(descriptionFT_wv,df['tkdescription']) # generate document embeddings


# In[47]:


GloVe_desc_dvs.head()


# In[48]:


# weighted
Weighted_GloVe_desc_dvs = gen_docVecs(descriptionFT_wv,df['tkdescription'],tfidf_weights)


# In[49]:


Weighted_GloVe_desc_dvs.head()


# ### 2.6.2 GloVe Check Unweighted and Weighted 

# In[50]:


GloVe_desc_dvs.isna().any().sum() # check whether there are any null values


# In[51]:


GloVe_desc_dvs.shape


# In[52]:


Weighted_GloVe_desc_dvs.isna().any().sum() # check whether there are any null values


# In[53]:


Weighted_GloVe_desc_dvs.shape


# ### 2.6.3 GloVe Visualisations

# In[54]:


#Weighted
features = GloVe_desc_dvs.to_numpy()
plotTSNE(df['jobCategory'],features)


# In[ ]:





# In[55]:


#Weighted
features = Weighted_GloVe_desc_dvs.to_numpy()
plotTSNE(df['jobCategory'],features)


# In[ ]:





# ## 2.7 Google News

# In[56]:


import gensim.downloader as api
GoogleNews_wv = api.load('word2vec-google-news-300')


# In[57]:


print(GoogleNews_wv)


# In[58]:


GoogleNews_wv.vector_size


# ### 2.7.1 Google Unweighted and Weighted

# In[59]:


desc_google_dvs = gen_docVecs(GoogleNews_wv,df['tkdescription'])


# In[60]:


desc_google_dvs.head()


# In[61]:


Weighted_desc_google_dvs = gen_docVecs(GoogleNews_wv,df['tkdescription'],tfidf_weights)


# In[62]:


Weighted_desc_google_dvs.head()


# ### 2.7.2 Google Check weighted and unweighted

# In[63]:


desc_google_dvs.shape


# In[64]:


desc_google_dvs.isna().any().sum()


# In[65]:


Weighted_desc_google_dvs.isna().any().sum()


# In[66]:


Weighted_desc_google_dvs.shape


# ### 2.7.3 Google Visualisations

# In[67]:


#UnWeighted
features = desc_google_dvs.to_numpy()
plotTSNE(df['jobCategory'],features)


# In[ ]:





# In[68]:


#Weighted
features = Weighted_desc_google_dvs.to_numpy()
plotTSNE(df['jobCategory'],features)


# In[ ]:





# ## Task 3. Job Advertisement Classification

# ### 3.1  - Q1 Language model comparisons

# In[69]:


# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score
from sklearn.linear_model import LogisticRegression
# Count_features - creating training and test split
X = count_features
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
count = predicted['test_score'].mean()

# FastText(Weighted) - creating training and test split
X = Weighted_descriptionFT_dvs
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
fastWeighted = predicted['test_score'].mean()

# FastText(Unweighted) - creating training and test split
X = descriptionFT_dvs
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
fastUnweighted = predicted['test_score'].mean()


# GloVe(Weighted)- creating training and test split
X = Weighted_GloVe_desc_dvs
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
gloveWeighted = predicted['test_score'].mean()

# GloVe(Unweighted) - creating training and test split
X = GloVe_desc_dvs
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
gloveUnweighted = predicted['test_score'].mean()


# GoogleNews(Weighted) - creating training and test split
X = Weighted_desc_google_dvs
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
googleWeighted = predicted['test_score'].mean()

# GoogleNews(Unweighted) - creating training and test split
X = desc_google_dvs
y = df['jobCategory'].tolist()

model = LogisticRegression()
predicted = cross_validate(model, X, y, cv=5)
googleUnweighted = predicted['test_score'].mean()


# In[70]:


dataf = [['Count_features', count], ['FastText (Weighted)', fastWeighted], ['FastText (Unweighted)', fastUnweighted],
        ['GloVe (Weighted)', gloveWeighted],['GloVe (Unweighted)', gloveUnweighted],['GoogleNews (Weighted)', googleWeighted],
        ['GoogleNews (Unweighted)', googleUnweighted]]
 
# Create the pandas DataFrame
dataf = pd.DataFrame(dataf, columns = ['Model', 'Predicted Model Score'])
 
# print dataframe.
dataf


# After computing 5-cross validation, The best performing model is `FastText(Weighted)` and `GloVe(Weighted` with a score of `0.874457` and the second best performing is count_features with a score of `0.873284`.

# 
# ### 3.2 - Question 2: Does more information provide higher accuracy?

# Performing the pre-processing of titles in the job advertisements.

# In[71]:


def tokenizeTitle(j_title):
   
    title = j_title.lower()  # convert all titles to lowercase
    # segmenting into sentences
    sentences = sent_tokenize(title)
    
    # tokenizing each sentence
    pattern=r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merging them into a list of tokens
    tokenised_title = list(chain.from_iterable(token_lists))
    return tokenised_title


# In[72]:


def stats_print(tkn_titles):
    words = list(chain.from_iterable(tkn_titles)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of titles:", len(tkn_titles))
    lens = [len(jtitle) for jtitle in tkn_titles]
    print("Average title length:", np.mean(lens))
    print("Maximun title length:", np.max(lens))
    print("Minimun title length:", np.min(lens))
    print("Standard deviation of title length:", np.std(lens))


# In[73]:


tkn_titles = [tokenizeTitle(w) for w in df['jobTitle']] 


# In[74]:


stats_print(tkn_titles)


# ### Task 3.2 Removing Single Character Token
# 
# Removing any tokens that contain single characters (a token that of less than length 2) in job titles. 
# Double checking whether it has been done properly.

# In[75]:


doubleChar_list = [[w for w in title if len(w) < 2]                       for title in tkn_titles] # create a list of single character token for each review
list(chain.from_iterable(doubleChar_list))


# In[76]:


tkn_titles[2]


# In[77]:


tkn_titles = [[w for w in title if len(w) >=2]                       for title in tkn_titles]


# #### The Statistics
# 
# After performing the tokenisation process, let's have a look at the statistics:

# In[78]:


stats_print(tkn_titles)


# ### Task 3.3 Removing Stop words in titles
# 
# Removing the stop words from the given `stopwords_en.txt`.

# In[79]:


Twords = list(chain.from_iterable(tkn_titles)) # we put all the tokens in the corpus in a single list
title_vocab = set(Twords) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words


# In[80]:


term_fd = FreqDist(Twords)


# In[81]:


stopwords_list = []
with open('stopwords_en.txt') as f:
    stopwords_list = f.read().splitlines()


# In[82]:


tkn_titles = [[w for w in title if w not in stopwords_list]                       for title in tkn_titles]


# In[83]:


Twords = list(chain.from_iterable([set(title) for title in tkn_titles]))
doc_fd = FreqDist(Twords)
doc_fd.most_common(25)


# In[84]:


rm_words = list(title_vocab - set(doc_fd.keys()))
print("Remove",len(rm_words), "number of stop words.")
rm_words


# In[85]:


print("Tokenized title:\n",tkn_titles[2])


# #### The Updated Statistics
# 
# After performing the tokenisation process, let's have a look at the statistics:

# In[86]:


stats_print(tkn_titles)


# ### 3.4 Removing Less Frequent Words i.e words that appear only once¶

# Removing the less frequent words from each tokenized title text by term frequency.
# 
# - find out the list of words that appear only once in the entire corpus of titles
# - remove these less frequent words from each tokenized title text

# In[87]:


Twords = list(chain.from_iterable(tkn_titles)) # we put all the tokens in the corpus in a single list


# In[88]:


lessFreqWords = set(term_fd.hapaxes())
lessFreqWords


# In[89]:


len(lessFreqWords)


# #### The Updated Statistics
# 
# In the above, we have done a few pre-processed steps, now let's have a look at the statistics again:
# We notice that the vocab size has reduced from ` ` to ``, a difference of ``.
# 
# 

# In[90]:


def removeLessFreqWords(title):
    return [d for d in title if d not in lessFreqWords]

tkn_titles = [removeLessFreqWords(title) for title in tkn_titles]


# In[91]:


stats_print(tkn_titles)


# ### 3.5 Constructing the Vocabulary

# In[92]:


# generating the vocabulary

Twords = list(chain.from_iterable(tkn_titles)) # we put all the tokens in the corpus in a single list
title_vocab = sorted(list(set(Twords))) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words

len(title_vocab)


# In[93]:


tkn_titles[2]


# In[94]:


#checking the length
len(title_vocab)


# In[95]:


len(job_title)


# ### 3.5 Generating Count Vector for Titles

# In[96]:


cVectorizer = CountVectorizer(analyzer = "word",vocabulary = title_vocab)
# initialised the CountVectorize


# In[97]:


count_features = cVectorizer.fit_transform([title for title in job_title]) # generate the count vector representation for all descriptions
print(count_features.shape)


# In[98]:


job_title[2]


# In[99]:


test_ind = 0


# In[100]:


def validator(data_features, title_vocab,a_ind,job_title):
    print("Title:",job_title[a_ind])
    print("--------------------------------------------\n")
    print("Vector representation:\n") # printing the vector representation as format 'word:value' (
                                      # the value is 0 or 1 in for binary vector; an integer for count vector; and a float value for tfidf
    for word, value in zip(title_vocab, data_features.toarray()[test_ind]): 
        if value > 0:
            print(word+":"+str(value), end =' ')


# In[101]:


validator(count_features,title_vocab,test_ind,job_title)


# In[102]:


count_features.shape


# ## Summary
# Give a short summary and anything you would like to talk about the assessment tasks here.

# In this task, task 2 has been completed with `count_vectors.txt`. And completed task 3 Q1. Due to time contraints, I coudn't finish Q2 of task 3. 
