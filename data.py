#%% 
# Load data in JSON files and other gendered data
import sys
sys.path.append('./')
import json
import logging

def import_all_data():
    # Gender pairs
    # For measures & this list, + = female, - = male
    logging.info("Loading all data from JSON and .txt files...")

    def_pairs = [('she','he'), ('her', 'his'), ('woman', 'man'),
                ('herself', 'himself'),('daughter', 'son'), ('mother', 'father'), 
                ('girl', 'boy'), ('female', 'male'), ('mary','john')]

    tested_female_profs = ["Nurse", "Hairdresser", "Secretary", "Dancer"]
    tested_male_profs = ["Doctor", "President", "Footballer", "Surgeon"]

    with open('data/words/professions.json', 'r') as f:
            professions = json.load(f)
    professions = [professions[i][0] for i in range(len(professions))] # list

    # Bolukbasi list of gender specific words       
    with open('data/words/gender_specific_full.json') as f:
        gender_specific = json.load(f)

    #  Bolukbasi long list of gender pairs
    with open('data/words/equalize_pairs.json') as f:
        equalize_pairs = json.load(f)
        
    # Test analogies file
    with open("data/analogies/word-test.v1.txt", 'r') as infile:
        analogs = infile.readlines()

    return def_pairs, tested_female_profs, tested_male_profs, professions, gender_specific, equalize_pairs, analogs
