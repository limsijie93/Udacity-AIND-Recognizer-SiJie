import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):

        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):

    def bic_score(self, n):
        """
            Return the bic score
        """
        model = self.base_model(n)

        logL = model.score(self.X, self.lengths)
        logN = np.log(len(self.X))

        # p = = n^2 + 2*d*n - 1
        d = model.n_features
        p = n ** 2 + 2 * d * n - 1

        return -2.0 * logL + p * logN, model

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        try:
            best_score = float("Inf") 
            best_model = None

            for n in range(self.min_n_components, self.max_n_components + 1):
                score, model = self.bic_score(n)
                if score < best_score:
                    best_score, best_model = score, model
            return best_model

        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):

    def calc_log_likelihood_other_words(self, model, other_words):
        return [model[1].score(word[0], word[1]) for word in other_words]

    def calc_best_score_dic(self, score_dics):
        return max(score_dics, key = lambda x: x[0])

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        other_words = []
        models = []
        score_dics = []
        for word in self.words:
            if word != self.this_word:
                other_words.append(self.hwords[word])
        try:
            for num_states in range(self.min_n_components, self.max_n_components + 1):
                hmm_model = self.base_model(num_states)
                log_likelihood_original_word = hmm_model.score(self.X, self.lengths)
                models.append((log_likelihood_original_word, hmm_model))
        except Exception as e:
            pass
        for index, model in enumerate(models):
            log_likelihood_original_word, hmm_model = model
            score_dic = log_likelihood_original_word - np.mean(self.calc_log_likelihood_other_words(model, other_words))
            score_dics.append(tuple([score_dic, model[1]]))
        return self.calc_best_score_dic(score_dics)[1] if score_dics else None


class SelectorCV(ModelSelector):

    def cv_score(self, n):

        scores = []
        split_method = KFold(n_splits=2)

        for train_idx, test_idx in split_method.split(self.sequences):
            self.X, self.lengths = combine_sequences(train_idx, self.sequences)

            model = self.base_model(n)
            test_X, test_l = combine_sequences(test_idx, self.sequences)

            scores.append(model.score(test_X, test_l))
        return np.mean(scores), model

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            best_score = float("Inf")
            best_model = None
            for n in range(self.min_n_components, self.max_n_components+1):
                score, model = self.cv_score(n)
                if score < best_score:
                    best_score = score
                    best_model = model
            return best_model
        except:
            return self.base_model(self.n_constant)
