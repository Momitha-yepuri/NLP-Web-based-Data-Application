#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 1. Basic Text Pre-processing
# #### Student Name: Momitha Yepuri
# #### Student ID: S3856512
# 
# Date: 3/10
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
# 
# In this assessment task, we are given a large collection of job advertisement documents (~ 50k jobs). The job advertisements range across 8 different industries. i.e `Accounting_Finance, Engineering, Healthcare_Nursing, Hospitality_Catering, IT, PR_Advertising_Marketing, Sales and Teaching`.
# The goal of this task is to perform basic text pre-processing on the job ads dataset, using processes such as tokenization, removing most/less frequent words and stop words, and extracting bigrams. Primarily, focusing on the pre-processing the description only.
# 
# We are outputting 3 text files,
# - `vocab.txt`: contains the unigram vocabulary, one each line, in the following format: `word_string:word_integer_index`.
# - `bigram.txt`: contains the found bigrams found in the whole document collection as well as their term frequency, separated by comma.
# - `job_ads.txt`: contains the job advertisement information and the pre-processed descriptiontext for all the job advertisement documents.
# 

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
import nltk.data
import re


# ## 1.1 Examining and loading data
# - Examine the data folder, including the categories and job advertisment txt documents, etc. Explain your findings here, e.g., number of folders and format of txt files, etc.
# - Load the data into proper data structures and get it ready for processing.
# - Extract webIndex and description into proper data structures.
# 

# The loaded `job_data` is then a dictionary, with the following attributes:
# * `data` - a list of job descriptions
# * `target` - the corresponding label of the job descriptions (integer index)
# * `target_names` - the names of job categories.
# * `filenames` - the filenames holding the dataset.

# In[2]:


ls data


# In[3]:


ls data/IT


# In[4]:


# Load the data from the data folder
job_data = load_files(r"./data")  


# In[5]:


job_data['filenames']


# In[6]:


job_data['target']


# In the data folder, there are 8 different subfolders where each folder is a job category.

# In[7]:


job_data['target_names']


# In[8]:


# test whether it matches, just in case
emp = 2
job_data['filenames'][emp], job_data['target'][emp] # from the file path we know that it's the correct class too


# In[9]:


#Assigning variables
descriptions, adverts = job_data.data, job_data.target  


# In[10]:


# description of job advertisement
descriptions[emp]


# In[11]:


adverts[emp]


# In[12]:


# getting the ID for each text file from each category
job_id = []
for id in job_data ['filenames']:
    job_id.append(id.split("Job_")[1].strip(".txt"))


# In[13]:


# creating a list for categories 
job_category = []
for cat1 in adverts:
    job_category.append(job_data['target_names'][cat1])


# ## 1.2 Pre-processing data

# ### 1.2.1 Tokenization

# In this sub-task, I'm tokenizing each of the job_ads description. First, converted the description into lowercase for consistency, then perform sentence segmentation followed by word tokenization. 
# Finally, Stored each tokenized description value as a list of tokens.

# In[14]:


#converting to lowercase
descriptions = [content.lower() for content in descriptions]


# In[15]:


# descriptions[emp]


# In[16]:


job_title = []
job_web_index = []
def tokenizeDescription(content):
   
    description = content.decode('utf-8') # convert the bytes-like object to python string, need this before we apply any pattern search on it
    #converting to lowercase
    to_lower = description.lower()
    
    #Searching for description using regex
    description = re.search(r'description:\s*(.*)$', str(to_lower)).group(1)
    
    #Searching for title using regex
    title = re.search(r'title:(.*)',str(to_lower)).group(1)
    title = title.strip() # strip whitespaces
    
    #Searching for webIndex using regex
    web_index = re.search(r'webindex:(.*\d+)',str(to_lower)).group(1)
    web_index = web_index.strip() # strip whitespaces
   
    # Storing all the results into a list after searcing job_title and webIndex through regex.
    job_title.append(title)
    job_web_index.append(web_index)
    #segmenting into sentences
    sentences = sent_tokenize(str(description))
    
    # tokenize each sentence
    pattern = r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"
    tokenizer = RegexpTokenizer(pattern) 
    token_lists = [tokenizer.tokenize(sen) for sen in sentences]
    
    # merge them into a list of tokens
    tokenised_description = list(chain.from_iterable(token_lists))
    return tokenised_description


# In[17]:


# test variable used throughout to test
test_ind = 2


# In[18]:


print("Number of Job ID's:", len(job_id))


# In[19]:


test_ind = 2 # randomly select an element to check whether the job ID and txt are correctly correspond to each other, 
print("Job ID:", job_id[test_ind])


# #### Statistics Before Any Further Pre-processing
# 
# * The total number of tokens across the corpus
# * The total number of types across the corpus, i.e. the size of vocabulary 
# * Lexical diversity referrs to the ratio of different unique word stems (types) to the total number of words (tokens).  
# * The average, minimum and maximum number of token (i.e. document length) in the dataset.
# 
# In the following, we are printing all these as a function, since we will use this printing module later to compare these statistic values before and after pre-processing.

# In[20]:


def stats_print(tk_descriptions):
    words = list(chain.from_iterable(tk_descriptions)) # we put all the tokens in the corpus in a single list
    vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words
    lexical_diversity = len(vocab)/len(words)
    print("Vocabulary size: ",len(vocab))
    print("Total number of tokens: ", len(words))
    print("Lexical diversity: ", lexical_diversity)
    print("Total number of descriptions:", len(tk_descriptions))
    lens = [len(desc) for desc in tk_descriptions]
    print("Average description length:", np.mean(lens))
    print("Maximun description length:", np.max(lens))
    print("Minimun description length:", np.min(lens))
    print("Standard deviation of description length:", np.std(lens))


# In[21]:


tk_descriptions = [tokenizeDescription(d) for d in descriptions]  # list comprehension, generate a list of tokenized descriptions


# In[22]:


print("Raw description:\n",descriptions[emp],'\n')
print("Tokenized description:\n",tk_descriptions[emp])


# #### The Statistics
# 
# After performing the tokenisation process, let's have a look at the statistics:

# In[23]:


stats_print(tk_descriptions)


# ### Task 1.2.2 Removing Single Character Token
# 
# Removing any tokens that contain single characters (a token that of less than length 2) in job descriptions. 
# Double checking whether it has been done properly.

# In[24]:


# create a list of single character token for each description
doubleChar_list = [[d for d in descriptions if len(d) < 2]                       for descriptions in tk_descriptions] 
list(chain.from_iterable(doubleChar_list)) # merge them together in one list


# In[25]:


# Before removal of 
print("Tokenized description:\n",tk_descriptions[emp])


# In[26]:


# filter out double character tokens
tk_descriptions = [[w for w in descriptions if len(w) >=2]                       for descriptions in tk_descriptions]


# In[27]:


# After removal
print("Tokenized description:\n",tk_descriptions[emp])


# In[28]:


stats_print(tk_descriptions)


# ### Task 1.2.3 Removing Stop words
# 
# Removing the stop words from the given `stopwords_en.txt`.

# In[29]:


# we put all the tokens in the corpus in a single list 
words = list(chain.from_iterable(tk_descriptions)) 
vocab = set(words) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words


# In[30]:


term_fd = FreqDist(words) # compute term frequency for each unique word/type


# In[31]:


stopwords_list = []
with open('stopwords_en.txt') as f:
    stopwords_list = f.read().splitlines()


# In[32]:


len(stopwords_list)


# In[33]:


tk_descriptions = [[w for w in description if w not in stopwords_list]                       for description in tk_descriptions]


# In[34]:


words = list(chain.from_iterable([set(description) for description in tk_descriptions]))
doc_fd = FreqDist(words)
doc_fd.most_common(25)


# In[35]:


rm_words = list(vocab - set(doc_fd.keys()))
print("Remove",len(rm_words), "number of stop words.")
rm_words


# In[36]:


print("Tokenized description:\n",tk_descriptions[emp])


# In[37]:


print("Before stopword removal:",len(descriptions),"tokens")
print("After stopword removal:",len(tk_descriptions),"tokens")


# #### The Updated Statistics
# 
# In the above, we have done a few pre-processed steps, now let's have a look at the statistics again:
# We notice that the vocab size has reduced from `89565` to `89052`, a difference of `513`.

# In[38]:


stats_print(tk_descriptions)


# ### Task 1.2.4 Removing Less Frequent Words i.e words that appear only once
# 
# Removing the less frequent words from each tokenized description text by term frequency.
# - find out the list of words that appear only once in the entire corpus of descriptions
# - remove these less frequent words from each tokenized description text

# In[39]:


words = list(chain.from_iterable(tk_descriptions)) # we put all the tokens in the corpus in a single list


# Finding out the set of less frequent words by using the `hapaxes` function applied on the **term frequency** dictionary. Hapaxes are words that occurs only once within a context.

# In[40]:


lessFreqWords = set(term_fd.hapaxes())
lessFreqWords


# In[41]:


#length of less frequenct words
len(lessFreqWords)


# We see that there are `48916` words that appear only once and we can prceed to remove them.

# In[42]:


def removeLessFreqWords(description):
    return [d for d in description if d not in lessFreqWords]

tk_descriptions = [removeLessFreqWords(description) for description in tk_descriptions]


# #### The Updated Statistics
# 
# In the above, we have done a few pre-processed steps, now let's have a look at the statistics again:
# We notice that the vocab size has reduced from `89052` to `40088`, a difference of `48964`.

# In[43]:


stats_print(tk_descriptions)


# ### Task 1.2.5 Removing the top 50 most frequent words based on document frequency.
# 
# 
# Removing the most frequent words from each tokenized description text. 
# Exploring the most frequent words in terms of document frequency:

# In[44]:


words = list(chain.from_iterable([set(description) for description in tk_descriptions]))
doc_fd = FreqDist(words)  # compute document frequency for each unique word/type
doc_fd.most_common(50)


# In[45]:


df_words = set(w[0] for w in doc_fd.most_common(50))
df_words


# In[46]:


# function to remove most frequent words in tk descriptions.
def removeMostFreqWords(description):
    return [d for d in description if d not in df_words]

tk_descriptions = [removeMostFreqWords(description) for description in tk_descriptions]


# #### The Updated Statistics
# 
# In the above, we have done a few pre-processed steps, now let's have a look at the statistics again:
# We notice that the vocab size has reduced from `40088` to `40038`, a difference of `50`.

# In[47]:


stats_print(tk_descriptions)


# ### Task 1.2.6 Extract the top 10 Bigrams based on term frequency
# Exploring the bigrams (top 10) in the pre-processed description text. Also making sense of the vocabulary.

# In[48]:


# adding all words 
words = list(chain.from_iterable(tk_descriptions))


# In[49]:


bigrams = ngrams(words, n = 2)
fdbigram = FreqDist(bigrams)


# In[50]:


bigrams = fdbigram.most_common(10) # top 10 bigrams
bigrams


# In[51]:


rep_patterns = [" ".join(bg[0]) for bg in bigrams]
rep_patterns


# In[52]:


replacements = [bg.replace(" ","_") for bg in rep_patterns] # convert the format of bigram into word1_word2
replacements


# #### The Updated Statistics
# 
# In the above, we have done a few pre-processed steps, now let's have a look at the statistics again. We notice that the vocab size has reduced from `40088` to `40038`, a difference of `50`.

# In[53]:


stats_print(tk_descriptions)


# ### 1.2.7 Constructing the Vocabulary

# In[54]:


# generating the vocabulary

words = list(chain.from_iterable(tk_descriptions)) # we put all the tokens in the corpus in a single list
vocab = sorted(list(set(words))) # compute the vocabulary by converting the list of words/tokens to a set, i.e., giving a set of unique words

len(vocab)


# In[55]:


tk_descriptions[test_ind]


# ## Saving Pre-processing required outputs
# Save the vocabulary, bigrams and job advertisment txt as per specification.
# - vocab.txt
# - bigram.txt
# - job_ads.txt
# 
# * unigram vocab saved in the following format: word_string:word_integer_index with the index value starts from 0. Stored in a .txt file named `vocab.txt`
#     * each line contains the unigram vocabulary
# * bigrams are based on their term frequency (from high to low) and store in a .txt file named `bigram.txt'
#     * contains the found bigrams found in the whole document collection as well as their term frequency, separated by comma (each line contains one bigram). 
# * 
#     
# Double Checked if this is saved properly.

# #### Saving the output for vocab

# In[56]:


out_file = open("vocab.txt", 'w') # creates a txt file named 'vocab.txt', open in write mode

for index in range(0, len(vocab)):
    out_file.write("{}:{}\n".format(vocab[index],index)) # write each index and vocabulary word, note that index start from 0
out_file.close() # close the file


# #### Saving the output for bigram

# In[57]:


out_file = open("bigram.txt", 'w') # creates a txt file named 'bigrams.txt', open in write mode
for word in bigrams:
    out_file.write(''.join(str(word)) + '\n') # join the tokens in an article with space, and write the obtained string to the txt document
out_file.close() # close the file


# #### Saving the output for job_ads

# In[58]:


out_file = open ("job_ads.txt", 'w')# creates a txt file named 'job_ads.txt', open in write mode

for i in range(len(job_id)):
    out_file.write("ID: {}\nCategory: {}\nWebindex: {}\nTitle: {}\nDescription: {}\n".
                   format(str(job_id[i]),str(job_category[i]),str(job_web_index[i]),
                          str(job_title[i]),str(" ".join(tk_descriptions[i]))))
    


# ## Summary
# Give a short summary and anything you would like to talk about the assessment task here.

# Basic Text Pre-processing on description has been successful.
