import numpy as np, matplotlib.pyplot as plt
from utils import Utils
from gensim.models import KeyedVectors
from scipy import spatial
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def test_default_bias(def_pairs, equalize_pairs, gender_specific):
    """
    Test bias for biased default word embeddings.

    @author Adriana Rodríguez Flórez
    @version December 2023
    """
    evaluator = GenderBiasEvaluator()
    evaluator.load_word_embeddings_google()
    return get_stats_for_we(evaluator, def_pairs, equalize_pairs, gender_specific)


def test_hard_debias(def_pairs, equalize_pairs, gender_specific, we_eval=None):
    """
    Executes hard debiasing on a given biased model.

    @author Adriana Rodríguez Flórez
    @version December 2003
    """
    if we_eval is None:
        we_eval =  GenderBiasEvaluator()
    biased_model = we_eval.load_word_embeddings_google() #keyed vectors (already loaded)

    vocab_limited, wv_vocab, vocab_neutral, wv_neutral, w2i_neutral, i2w_neutral = Utils.limit_vocab("we", biased_model, 
                                                                                                    Utils.excluded_words(def_pairs, equalize_pairs, gender_specific))
    
    print("Procceeding to hard-debias...")
    we_eval.hard_debias(vocab_limited, wv_vocab, gender_specific, def_pairs, equalize_pairs) # debiased model
    return get_stats_for_we(we_eval, def_pairs, equalize_pairs, gender_specific)

def test_double_hard_debias(def_pairs, equalize_pairs, gender_specific, we_eval=None):
    """
    Executes hard debiasing on a given biased model.

    @author Adriana Rodríguez Flórez
    @version December 2003
    """
    if we_eval is None:
        we_eval = GenderBiasEvaluator()
    biased_model = we_eval.load_word_embeddings_google() #keyed vectors (already loaded)

    vocab_limited, wv_vocab, vocab_neutral, wv_neutral, w2i_neutral, i2w_neutral = Utils.limit_vocab("we", biased_model, 
                                                                                                    Utils.excluded_words(def_pairs, equalize_pairs, gender_specific))
    logging.info("Procceeding to double hard-debias...")
    we_eval.double_hard_debias(vocab_limited, wv_vocab, gender_specific, def_pairs, equalize_pairs) # debiased model
    return get_stats_for_we(we_eval, def_pairs, equalize_pairs, gender_specific, vocab_limited=vocab_limited)



def get_stats_for_we(evaluator, def_pairs, equalize_pairs, gender_specific, vocab_limited=None, method_type="we"):
    """
    Testing gender bias in word embeddings.

    @author Adriana Rodríguez Flórez
    @version December 2023
    """

    # Limit the vocabulary, and neutralize it
    logging.info("Limiting word vocabulary and indexing words and vectors...")
    if vocab_limited is None:
        vocab_limited, wv_vocab, vocab_neutral, wv_neutral, w2i_neutral, i2w_neutral = Utils.limit_vocab(method_type, evaluator.model, 
                                                                                                    Utils.excluded_words(def_pairs, equalize_pairs, gender_specific))
    # Compute all means and std. deviations
    # The dictionary has defined pairs (i.e. [he,she] and the respective bias computations)
    logging.info("Computing bias statistics for the word embeddings...")
    mean, std, dictionary = evaluator.direct_bias_stats(method_type, evaluator.model, vocab_limited, def_pairs)    

    return evaluator, evaluator.model, mean, std, dictionary


def plot_results_for_we(evaluator, model, mean, std, def_pairs, tested_female_profs, tested_male_profs, method_type="we"):
    # PLOT: for desired occupations
    logging.info("Plotting bias statistics for the word embeddings (w.r.t. to gender-specific pairs)...")
    plot_for(method_type, evaluator, model, tested_female_profs, mean, std, def_pairs)
    plot_for(method_type, evaluator, model, tested_male_profs, mean, std, def_pairs)


# Plotting results
def plot_for(method_type, evaluator, model, professions, means, std, def_pairs):
    first = True
    for prof in professions:
        # Get different variations for the same word (singular, plural, uppercase, lowercase...)
        # as long as they are found in the model's vocabulary
        prof_words = Utils.get_word_variations(method_type, model, prof)

        if (first): # Only show legend for first example
            evaluator.plot_gender_bias(model, method_type, prof, prof_words, means, std, def_pairs, leg=True)
            first = False
        else:
            evaluator.plot_gender_bias(model, method_type, prof, prof_words, means, std, def_pairs)



class GenderBiasEvaluator:

    """    
    This class contains the implementation of Word2Vec word embeddings to display
    specific gender bias based on sets of pre-defined words.

        Based on the paper "Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings"
    by Tolga Bolukbasi and Kai-Wei Chang and James Zou and Venkatesh Saligrama and Adam Kalai,

        and

        "Robustness and Reliability of Gender Bias Assessment in Word Embeddings: The Role of Base Pairs",
    by Haiyang Zhang, Alison Sneyd and Mark Stevenson, AACL 2020

        and

        Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013)."Efficient estimation of word representations in vector space"

    @author Adriana Rodríguez Flórez
    @version November 2023
    """

    def __init__(self):
        self.model = None


    def load_word_embeddings_google(self):
        """
        Loads the pre-existing vectorization found in the Google News dataset
        (billions of words analyzed).
        Needs to be loaded only once.

        @return model and normalized model
        """
        # load word embeddings from the basic Word2Vec Google News embeddings file to test
        if self.model is None:
            logging.info("Loading Google word embeddings...")
            self.model = KeyedVectors.load_word2vec_format('data/embeddings/GoogleNews-vectors.bin', binary=True)
            self.model.save("results/default/google-model.kv")
        return self.model
    
    #
    # ---- BOLUKBASI gender bias metrics
    #

    def direct_bias(self,method_type, model,word,pair):
        """
        Definition of Bolukbasi direct bias metrics (Caliskan word association with one base pair).

        @param model trained on word embeddings
        @param word to analyze
        @pair to which it is being analyzed ('he/she', 'man/woman')
        """
        try:
            if (method_type == 'we'):
                word = model[word]/np.linalg.norm(model[word]) 
                pair0 = model[pair[0]]/np.linalg.norm(model[pair[0]]) # all pair words in vocab
                pair1 = model[pair[1]]/np.linalg.norm(model[pair[1]]) 
                db = np.dot(word, pair0-pair1)
            elif (method_type == 'cda'):
                word = model.wv.get_vector(word)/np.linalg.norm(model.wv.get_vector(word)) 
                pair0 = model.wv.get_vector(pair[0])/np.linalg.norm(model.wv.get_vector(pair[0]))
                pair1 = model.wv.get_vector(pair[1])/np.linalg.norm(model.wv.get_vector(pair[1]))
                db = np.dot(word, pair0-pair1)
            return db
        except:
            return 0


    def direct_bias_stats(self, method_type, model, vocab_limited, def_pairs):
        """
        Statistical computation of arithmetic average + standard deviation
        for the direct bias scores, and the pair-bias computations for the dictionary.

        @return mean, std arithmetic average and standard deviation, and dictionary of pairs and bias to words
        """
        pairs_bias = {}
        all_scores =[]

        for pair in def_pairs:
            pairs_bias[pair] = []

            for word in vocab_limited:
                bias_value = self.direct_bias(method_type, model, word, pair)
                pairs_bias[pair].append([word,bias_value])
                all_scores.append(bias_value)
                
        mean = np.mean(all_scores)
        std = np.std(all_scores)
        
        print("Bolukbasi (direct bias) mean: ", mean)
        print("Bolukbasi (direct bias) std.:", std)

        return mean, std, pairs_bias


    # Function to plot scores of word list for a given measure and list of base pairs
    def plot_gender_bias(self, model, method_type, word, gword_list, mean, std, pair_list, leg = False):
        """
        Plots scores of a given word list (different variations of the same word in terms
        of number, upper/lowercase...), based on the mean and standard deviation,
        in terms of a given list of gendered pairs.

        @param word to analyze
        @param std, mean standard deviation and arithmetic average
        @param pair_list list of gendered opposite pairs
        @param gword_list list of variations of the same word
        """

        length = len(pair_list)
        x = np.arange(length)
        
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("Gender bias for: " + word.capitalize())
        graph_types = ["-o", "-^", "-*", "-D"]
        
        for i,word in enumerate(gword_list):
            graph_type = graph_types[i]

            scores = [self.direct_bias(method_type, model, word, pair) for pair in pair_list]
            if (sum(scores) != 0):
                plt.plot(x, scores, graph_type, label = word)

        plt.plot(x, [mean for i in range(length)], "g--", label = "mean")
        plt.plot(x, [mean-std for i in range(length)], "-.",color = "orange", label = "1 standard deviation")
        plt.plot(x, [mean+std for i in range(length)],  "-.",color = "orange", label =  "")
        plt.plot(x, [mean-2*std for i in range(length)],  "r:", label = "2 standard deviations")
        plt.plot(x, [mean+2*std for i in range(length)],  "r:", label = "")
        plt.xticks(x, pair_list, rotation='vertical')

        if leg == True:
            plt.legend()
            ax.legend(bbox_to_anchor=(1.05, 1.0))
        plt.show()

    
    def cosine_similarity(self, v1, v2):
        """
        Computation of the cosine similarity between two word-vectors.
            i.e. v1 = model['king]; v2 = model['queen']

        @params v1,v2 word vectors found in the embedding
        """
        return 1 - spatial.distance.cosine(v1, v2)



    #
    # --- DEBIASING METHOD 1: Hard-debias
    # Tolga Bolukbasi et al.
    #

    def hard_debias(self, limited_vocab, wv_vocab, gender_specific, def_pairs, equalized_pairs):
        """
        Executes direct, hard debiasing on a given model
        by means of normalizing all the vectors,
        so that each gendered word is at the same distance from
        one gender opposite to the other.

        @param word embeddings to debias
        """
        words = {}
        indexes = {}
        vectors = []
        i = 0

        logging.info("Indexing words and vectors for hard-debias...")
        for i in range(0,len(wv_vocab)):
            words[i] = limited_vocab[i] # save the word
            indexes[words[i]] = i # save the index of that word
            vectors.append(wv_vocab[i]) # save the vector of that word

        logging.info("Computing principal component analysis to infer gender direction...")
        gender_direction = Utils.doPCA(def_pairs, words, vectors, indexes).components_[0]
        specific_set = set(gender_specific)

        for i, w in enumerate(words):
            if w not in specific_set:
                vectors[i] = Utils.drop(vectors[i], gender_direction)

        logging.info("Normalizing and equalizing all vectors...")
        vectors = Utils.normalize(vectors)
        candidates = {x for e1, e2 in equalized_pairs for x in [(e1.lower(), e2.lower()),
                                                        (e1.title(), e2.title()),
                                                        (e1.upper(), e2.upper())]}
        for (a, b) in candidates:
            if (a in words.values() and b in words.values()):
                y = Utils.drop((vectors[indexes[a]] + vectors[indexes[b]]) / 2, gender_direction)
                z = np.sqrt(1 - np.linalg.norm(y)**2)

                if (vectors[indexes[a]] - vectors[indexes[b]]).dot(gender_direction) < 0:
                    z = -z

                vectors[indexes[a]] = z * gender_direction + y
                vectors[indexes[b]] = -z * gender_direction + y

        normalized_vectors = Utils.normalize(vectors)

        # We can now create a new, debiased model
        logging.info("Word embeddings successfully debiased (HARD-DEBIAS)!")
        debiased_model = KeyedVectors(vector_size=300)
        debiased_model.add_vectors(words, normalized_vectors, replace=True)

        self.model = debiased_model
        debiased_model.save("results/hard_debias/hb-debiased-model.kv")
        return debiased_model


    #
    # --- DEBIASING METHOD 2: Double Hard-debias
    # Tianlu Wang et al.
    #

    def double_hard_debias(self, limited_vocab, wv_vocab, gender_specific, def_pairs, equalized_pairs):
        """
        Executes double hard debiasing on a given model.
        Due to the huge size of the dataset, this is only executed on the limited vocabulary.
        Hard debias is improved by taking into account the word frequency
        which influences on the outcome of the hard-debias algorithm itself.

        @param word embeddings to debias
        """
        words = {}
        indexes = {}
        vectors = []
        i = 0

        logging.info("Indexing words, frequencies and vectors for double hard-debias...")
        for i in range(0,len(wv_vocab)):
            words[i] = limited_vocab[i] # save the word
            indexes[words[i]] = i # save the index of that word
            vectors.append(wv_vocab[i]) # save the vector of that word

        logging.info("Ignoring all frequencies and analyzing principal components...")
        main_pca = Utils.doMainPCA(def_pairs, vectors, indexes) # PCA
        D = []; D.extend(main_pca.components_)

        wv_mean = np.mean(vectors)
        for i, w in enumerate(words):
            u = wv_vocab[i, :]
            sub = np.zeros(u.shape).astype(float)
            for d in D:
                sub += np.dot(np.dot(np.transpose(d), u), d)
            wv_vocab[i, :] = wv_vocab[i, :] - sub - wv_mean


        logging.info("Computing principal component analysis to infer gender direction...")
        gender_direction = Utils.doPCA(def_pairs, vectors, indexes).components_[0]
        specific_set = set(gender_specific)

        for i, w in enumerate(words):
            if w not in specific_set:
                vectors[i] = Utils.drop(vectors[i], gender_direction)


        logging.info("Normalizing and equalizing all vectors...")
        vectors = Utils.normalize(vectors)
        candidates = {x for e1, e2 in equalized_pairs for x in [(e1.lower(), e2.lower()),
                                                        (e1.title(), e2.title()),
                                                        (e1.upper(), e2.upper())]}
        for (a, b) in candidates:
            if (a in words.values() and b in words.values()):
                y = Utils.drop((vectors[indexes[a]] + vectors[indexes[b]]) / 2, gender_direction)
                z = np.sqrt(1 - np.linalg.norm(y)**2)

                if (vectors[indexes[a]] - vectors[indexes[b]]).dot(gender_direction) < 0:
                    z = -z

                vectors[indexes[a]] = z * gender_direction + y
                vectors[indexes[b]] = -z * gender_direction + y

        normalized_vectors = Utils.normalize(vectors)

        # We can now create a new, debiased model
        logging.info("Word embeddings successfully debiased (DOUBLE HARD-DEBIAS)!")
        debiased_model = KeyedVectors(vector_size=300)
        debiased_model.add_vectors(words, normalized_vectors, replace=True)

        self.model = debiased_model
        debiased_model.save("results/hard_debias_2/double-hb-debiased-model.kv")
        return debiased_model
