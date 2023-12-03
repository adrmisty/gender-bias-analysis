#%%
# 0. Import all necessary data
import sys
sys.path.append('./')
from data import import_all_data
from gender_bias_evaluator import plot_results_for_we

def_pairs, tested_female_profs, tested_male_profs, professions, gender_specific, equalize_pairs, analogs = import_all_data()

#%%
# 1. Default word embeddings
from gender_bias_evaluator import test_default_bias
we_eval, biased_we, biased_mean, biased_std, dictionary = test_default_bias(def_pairs, equalize_pairs, gender_specific)

plot_results_for_we(we_eval, biased_we, biased_mean, biased_std, def_pairs, tested_female_profs, tested_male_profs)

#%%
# 2. Hard, direct debiasing (equalizing + normalizing)
from gender_bias_evaluator import test_hard_debias
unbiased_we_eval, unbiased_we, unbiased_mean, unbiased_std, unbiased_dictionary = test_hard_debias(def_pairs, equalize_pairs, gender_specific, we_eval=None)
plot_results_for_we(unbiased_we_eval, unbiased_we, unbiased_mean, unbiased_std, def_pairs, tested_female_profs, tested_male_profs)

#%%
# 3. Double hard-debias (no frequencies)
from gender_bias_evaluator import test_double_hard_debias
unbiased_we_eval_2, unbiased_we_2, unbiased_mean_2, unbiased_std_2, unbiased_dictionary_2 = test_double_hard_debias(def_pairs, equalize_pairs, gender_specific, we_eval=None)
plot_results_for_we(unbiased_we_eval_2, unbiased_we_2, unbiased_mean_2, unbiased_std_2, def_pairs, tested_female_profs, tested_male_profs)

#%%
# 4. CDA word embeddings (dataset augmentation)
from cda_substitution import test_cda_word_embeddings
cda_we_eval, cda_we, cda_mean, cda_std, cda_dictionary = test_cda_word_embeddings(tested_male_profs, tested_female_profs, def_pairs, equalize_pairs, gender_specific)
plot_results_for_we(cda_we_eval, cda_we, cda_mean, cda_std, def_pairs, tested_female_profs, tested_male_profs)

#%%
