import json, re, numpy as np
from os import walk
import codecs
from tqdm import tqdm
from sklearn.decomposition import PCA
import logging
from numpy import linalg as LA
from tqdm import tqdm
import string

class Utils:
    """
    Definitions of static, utility functions for:
        - JSON processing
        - Text processing
        - File processing
        - Numpy processing
        - Word2Vec processing
    """

    @staticmethod
    def save(inpath,outpath):
        outfile = open(outpath,'w')
        for folder , sub_folders , files in walk(inpath):
            for f in files:
                current_path = folder+"\\"+f
                current_file = open(current_path, 'r')
                for line in current_file.readlines():
                    outfile.write(line)
                #close the file
        #close your output file
        outfile.close()

    @staticmethod
    def load_json_pairs(input_file):
        with open(input_file, "r") as fp:
            pairs = json.load(fp)
        return pairs
    
    @staticmethod
    def get_word_variations(method_type, model, word):
        """
        @return different variations of a word, in terms of number
        """
        word = word.lower() # all words in the vocab are inherently lowercase
        words = [word]

        if method_type == "cda":
            if model.wv.has_index_for(word+"s"):
                words.append(word+"s")
            if model.wv.has_index_for(word+"es"):
                words.append(word+"es")
            if model.wv.has_index_for(word[:-1]+"ies"):
                words.append(word[:-1]+"ies")

        elif method_type == "we":
            try:
                model[word+'s']
                words.append(word+"s")
            except:
                pass

            try:
                model[word+'es']
                words.append(word+"es")
            except:
                pass

            try:
                model[word[:-1]+'ies']
                words.append(word+"ies")
            except:
                pass
            
        return words
        
    @staticmethod
    def normalize(vectors):
        vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        return vectors

    @staticmethod
    def drop(u, v):
        return u - v * u.dot(v) / v.dot(v)

    @staticmethod
    def safe_word(w):
        # ignore words with numbers, etc.
        # [a-zA-Z\.'_\- :;\(\)\]] for emoticons
        return (re.match(r"^[a-zA-Z_]*$", w) and len(w) < 20 and not re.match(r"^_*$", w))

    @staticmethod
    def neutral_word(w):
        # ignore words with numbers, etc.
        # [a-zA-Z\.'_\- :;\(\)\]] for emoticons
        return (re.match(r"^[a-z_]*$", w) and len(w) < 20 and not re.match(r"^_*$", w))

    @staticmethod
    def excluded_words(def_pairs, equalize_pairs, gender_specific):
        """
        @return list of excluded words based on defined pairs, and gender-specific words
        """
        exclude_words = []
        for pair in def_pairs + equalize_pairs:
            exclude_words.append(pair[0])
            exclude_words.append(pair[1])
        exclude_words = list(set(exclude_words).union(set(gender_specific)))

    @staticmethod
    def doPCA(pairs, vectors, indexes, num_components = 10):
        matrix = []
        
        for a, b in pairs:
            center = (vectors[indexes[a]] + vectors[indexes[b]])/2
            matrix.append(vectors[indexes[a]] - center)
            matrix.append(vectors[indexes[b]] - center)
        matrix = np.array(matrix)
        
        pca = PCA(n_components = num_components)
        pca.fit(matrix)
        # bar(range(num_components), pca.explained_variance_ratio_)
        return pca


    @staticmethod
    def doMainPCA(pairs, vectors, indexes, num_components = 10):
        logging.info("Fitting model with principal component analysis for double hard-debias...")
        wv_mean = np.mean(np.array(vectors), axis=0)
        matrix = []

        for a,b in pairs:
            matrix.append(vectors[indexes[a]] - wv_mean)
            matrix.append(vectors[indexes[b]] - wv_mean)
        matrix = np.array(matrix)

        main_pca = PCA(n_components = num_components)
        main_pca.fit(matrix)        
        return main_pca


    @classmethod
    def limit_vocab(cls, method_type, model, exclude = None):
        vocab_limited = []
        vocab_neutral = []

        # Iterate over all words
        if (method_type == "cda"):
            for w in model.wv.index_to_key [:len(model.wv)]:
                if (isinstance(w,str)): # Avoid True/False string errors
                        for word in w: # Iterate over all words
                            if (isinstance(word,str)): # Avoid True/False string errors
                                if cls.neutral_word(word) == True:
                                    vocab_limited.append(word)
                                    vocab_neutral.append(word)
                                    
        # Keyed vectors
        elif (method_type == "we"):
            for w in model.index_to_key [:len(model)]:
                    if cls.neutral_word(w) == True:
                        vocab_limited.append(w)
                        vocab_neutral.append(w)

        if exclude:
            vocab_neutral = list(set(vocab_neutral) - set(exclude))
    
        print("Size of limited vocabulary: ", len(vocab_limited))

        wv_vocab = np.zeros((len(vocab_limited), 300))
        for i,w in enumerate(vocab_limited):
            if (method_type == "cda"):
                wv_vocab[i,:] = model.wv.get_vector(w)
            elif (method_type == "we"):
                wv_vocab[i,:] = model[w]
        
        wv_neutral = np.zeros((len(vocab_neutral), 300))
        for i,w in enumerate(vocab_neutral):
            if (method_type == "cda"):
                wv_neutral[i,:] = model.wv.get_vector(w)
            elif (method_type == "we"):
                wv_neutral[i,:] = model[w]
            
        # word2index/index2word
        w2i_neutral = {w: i for i, w in enumerate(vocab_neutral)}
        i2w_neutral = {i:w for w, i in w2i_neutral.items()}
        
        return vocab_limited, wv_vocab, vocab_neutral, wv_neutral, w2i_neutral, i2w_neutral
    


class TwoWayDict(dict):
    """
    Implementation of a Two-Way dictionary.
    """
    def __setitem__(self, key, value):
        # Remove any previous connections with these values
        if key in self:
            del self[key]
        if value in self:
            del self[value]
        dict.__setitem__(self, key, value)
        dict.__setitem__(self, value, key)

    def __delitem__(self, key):
        dict.__delitem__(self, self[key])
        dict.__delitem__(self, key)

    def __len__(self):
        """Returns the number of connections"""
        return dict.__len__(self) // 2    