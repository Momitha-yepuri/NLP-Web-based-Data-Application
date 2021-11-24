import numpy as np
import pandas as pd

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




# def gen_docVecs(wv,tk_txts): # generate vector representation for documents
#     docs_vectors = pd.DataFrame() # creating empty final dataframe
#     #stopwords = nltk.corpus.stopwords.words('english') # if we haven't pre-processed the articles, it's a good idea to remove stop words

#     for i in range(0,len(tk_txts)):
#         tokens = tk_txts[i]
#         temp = pd.DataFrame()  # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
#         for w_ind in range(0, len(tokens)): # looping through each word of a single document and spliting through space
#             try:
#                 word = tokens[w_ind]
#                 word_vec = wv[word] # if word is present in embeddings(goole provides weights associate with words(300)) then proceed
#                 temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
#             except:
#                 pass
#         doc_vector = temp.sum() # take the sum of each column
#         docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
#     return docs_vectors


