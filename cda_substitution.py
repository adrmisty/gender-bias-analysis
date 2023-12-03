import sys
sys.path.append('./')
import random
import spacy
from utils import Utils, TwoWayDict
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import re
from multiprocessing import Pool
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gender_bias_evaluator import GenderBiasEvaluator, get_stats_for_we


def test_cda_word_embeddings(tested_male_profs, tested_female_profs, def_pairs, equalize_pairs, gender_specific):
    """
    Testing gender bias in word embeddings after having applied CDA and CDS.

    @author Adriana Rodríguez Flórez
    @version December 2003
    """

    cda_eval = GenderBiasEvaluator_CDA()
    model_cda = cda_eval.generate_cda_model()
    return get_stats_for_we(cda_eval, def_pairs, equalize_pairs, gender_specific, method_type="cda")



class GenderBiasEvaluator_CDA(GenderBiasEvaluator):

    """
    'Gender Bias Evaluator' for CS4824 - Machine Learning final project.
    Subclass of GenderBiasEvaluator - contains implementation of Counterfactual Data Augmentation
    with Counterfactual Data Substitution for a given dataset.

    @author Adriana Rodríguez Flórez
    @version November 2023
    """

    def __init__(self):
        super().__init__ 
        self.model = None


    def generate_word_embeddings(self, dataset, name, text_file="results/augmented.txt"):
        """
        Generates a new World2Vec embedding based on a dataset that we provide.

        @param list of texts
        @param name of the newly generated model
        @return model based on word embeddings
        """
        sentences = [word_tokenize(sent) for sent in dataset]
        model = Word2Vec(sentences = sentences, vector_size = 300, sg = 1, window = 3, min_count = 1, epochs = 10, workers = Pool()._processes)
        model.init_sims(replace = True)
        model.save('results/cda/model-' + name)

        return model
    

    ###
    ### --- Apply Counterfactual Data Augmentation on a dataset
    ###

    def generate_cda_model(self,infile="results/cda/full-corpus.txt", file="data/coca", outfile="results/augmented.txt", num_texts=4378):
        """
        For a textual dataset of thousands of sentences,
        name-based Counterfactual Data Augmentation with Counterfactual Data Substitution
        is executed. A new dataset is generated, from which word embeddings will be obtained
        thus creating a new model.

        @param input text file to read from
        @return model trained on word-embeddings of a counterfactually substituted/augmented dataset
        """
        
        already_generated = True
        if already_generated:
            # Uses: corpus of contemporary American English --> https://www.english-corpora.org/coca/
            # NOTE: Already been generated, no need to uncomment this line (takes too long...)
            #model_cda = cda_eval.generate_cda_model("data/coca/all.txt")
            logging.info("Loading pre-trained model on counterfactually-augmented CODA corpus...")
            model_cda = Word2Vec.load("results/cda/model-cda") # Use pre-generated model (generating takes TOO LONG)
            self.model = model_cda
            return model_cda
        else:
            # Takes very long
            new_sentences = []
            logging.info("Starting CDA/CDS process on the input dataset...")
            base_pairs = Utils.load_json_pairs('data/words/cda_default_pairs.json')
            name_pairs = Utils.load_json_pairs('data/words/names_pairs_1000_scaled.json')
            substitutor = CDA_Substitutor(base_pairs, name_pairs=name_pairs)

            logging.info("Reading input corpus file...")
            Utils.save(file, infile)
            logging.info("Full analyzed corpus available at: " + infile)
            logging.info("Starting probabilistic Counterfactual Data Substitution...")
            f = open(infile, "r")
            fw = open(outfile, "w")
            i = 1

            for line in f.readlines():
                
                if (i == num_texts // 4):
                    logging.info("25% processed...")
                elif (i == num_texts // 2):
                    logging.info("50% processed...")
                elif (i == num_texts*3 // 4):
                    logging.info("75% processed...")

                words = re.findall(r'\w+', line)
                text = [word.lower() for word in words]
                lowercased = " ".join(text)

                # Intervene probablistically (50% of sentences will be flipped)
                flipped,text = substitutor.probablistic_substitute(lowercased)

                # Save the words
                new_sentences.append(text)
                fw.write(text + "\n")
                i += 1

            f.close()
            fw.close()
            logging.info("Generating word embeddings for the newly augmented dataset...")
            return self.generate_word_embeddings(new_sentences, "cda")
        


class CDA_Substitutor:
    """
    Implementation of a Counterfactual Data Substitution algorithm.
    Substitutes all gendered words for their opposite gender,
    it can do it:
        - linearly
        - probabilistically (will do it at random, on 50% of the dataset)

    @author https://github.com/rowanhm/counterfactual-data-substitution
    """

    def __init__(self, base_pairs, name_pairs=None, his_him=True, spacy_model='en_core_web_lg'):

        logging.warn("For this to execute correctly, run first: python -m spacy download en_core_web_lg")
        logging.info("Loading default Spacy model...")
        # python -m spacy download en_core_web_lg
        self.nlp = spacy.load(spacy_model)
        logging.info("Default Spacy model loaded successfully.")

        # This flag tells it whether or not to apply the special case intervention to him/his/her/hers
        self.his_him = his_him

        self.base_pairs = TwoWayDict()
        for (male, female) in base_pairs:
            self.base_pairs[male.lower()] = female.lower()

        self.name_pairs = TwoWayDict()
        for (male, female) in name_pairs:
            self.name_pairs[male.lower()] = female.lower()


    def probablistic_substitute(self, text):
        if bool(random.getrandbits(1)):
            return True, self.invert_document(text)
        else:
            return False, text


    def invert_document(self, input_text):
        # Parse the doc
        doc = self.nlp(input_text)

        output = input_text

        # Walk through in reverse order making substitutions
        for word in reversed(doc):

            # Calculate inversion
            flipped = self.invert_word(word)

            if flipped is not None:
                # Splice it into output
                start_index = word.idx
                end_index = start_index + len(word.text)
                output = output[:start_index] + flipped + output[end_index:]

        return output

    def invert_word(self, spacy_word):

        flipped = None
        text = spacy_word.text.lower()

        # Handle base case
        if text in self.base_pairs.keys():
            flipped = self.base_pairs[text]

        # Handle name case
        elif text in self.name_pairs.keys() and spacy_word.ent_type_ == "PERSON":
            flipped = self.name_pairs[text]

        # Handle special case (his/his/her/hers)
        elif self.his_him:
            pos = spacy_word.tag_
            if text == "him":
                flipped = "her"
            elif text == "his":
                if pos == "NNS":
                    flipped = "hers"
                else:  # PRP/PRP$
                    flipped = "her"
            elif text == "her":
                if pos == "PRP$":
                    flipped = "his"
                else:  # PRP
                    flipped = "him"
            elif text == "hers":
                flipped = "his"

        if flipped is not None:
            # Attempt to approximate case-matching
            return self.match_case(flipped, spacy_word.text)
        return None

    @staticmethod
    def match_case(input_string, target_string):
        # Matches the case of a target string to an input string
        # This is a very naive approach, but for most purposes it should be okay.
        if target_string.islower():
            return input_string.lower()
        elif target_string.isupper():
            return input_string.upper()
        elif target_string[0].isupper() and target_string[1:].islower():
            return input_string[0].upper() + input_string[1:].lower()
        else:
            logging.warning("Unable to match case of {}".format(target_string))
            return input_string
