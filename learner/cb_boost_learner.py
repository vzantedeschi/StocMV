import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from voter.stump import DecisionStumpMV
from voter.majority_vote import MajorityVote


class CBBoostLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, majority_vote, n_max_iterations=100, twice_the_same=True):
        super(CBBoostLearner, self).__init__()

        self.mv = majority_vote
        assert isinstance(self.mv, MajorityVote)
        assert self.mv.fitted
        assert self.mv.complemented

        self.quasi_uniform = self.mv.quasi_uniform
        if(self.quasi_uniform):
            self.mv.quasi_uniform_to_normal()

        self.n_max_iterations = n_max_iterations
        self.twice_the_same = twice_the_same

    def fit(self, X, y):
        # X -> (size, nb_feature)
        # y -> (size, 1)
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(X.shape) == 2 and len(y.shape) == 2 and
                X.shape[0] == y.shape[0] and
                y.shape[1] == 1 and X.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        # We compute the output matrix
        out = self.mv.output(X)

        # We initialize the list of selected voters
        self.selected_voter_list = []

        # We compute the individual margin
        y_out_matrix = y*out

        # Get n (the number of voters)
        n = y_out_matrix.shape[1]

        # We initialize the posterior
        self.post = np.zeros(self.mv.post.shape)

        # Initialize the majority vote
        self.get_first_voter(y_out_matrix)

        # For each iteration
        it = n-1
        if(self.n_max_iterations is not None and self.n_max_iterations < n-1):
            it = self.n_max_iterations - 1
        for k in range(it):

            # We get a new voter with its weight (and stop otherwise)
            if(not(self.get_new_voter(y, out, y_out_matrix))):
                break

        # We normalize (and update) the weights
        self.post = self.post/np.sum(self.post)
        self.mv.post = self.post

        if(self.quasi_uniform):
            self.mv.normal_to_quasi_uniform()

        return self

    def get_first_voter(self, y_out_matrix):

        # We compute the margin
        margin_matrix = np.sum(y_out_matrix, axis=0)

        # We get the voter with the highest margin (and set its weight to 1)
        margin_argmax = np.argmax(margin_matrix)
        self.selected_voter_list.append(margin_argmax)
        self.post[margin_argmax] = 1.0


    def get_new_voter(self, y, out, y_out_matrix):
        # We apply Theorem 3 to get the new voter
        m = y_out_matrix.shape[0]

        # We compute F_k
        F_k = out@self.post

        # We compute Definitions 8, 9 and 10 for h_k and F_k
        tau_F_k_h_k = np.mean(F_k*out, axis=0)
        gamma_F_k = np.mean(y*F_k, axis=0)
        gamma_h_k = np.mean(y_out_matrix, axis=0)
        mu_F_k = np.mean(F_k**2.0, axis=0)

        # We generate the test for the optimal alpha_k
        test_alpha = (tau_F_k_h_k-gamma_F_k/gamma_h_k)

        # We compute the alpha_k optimal when test_alpha < 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_k = ((gamma_h_k*mu_F_k - gamma_F_k*tau_F_k_h_k)
                       / (gamma_F_k-gamma_h_k*tau_F_k_h_k))

        # We intialize the C-Bounds with the different alpha_k
        # (that respect the conditions)
        c_bound_k = np.zeros(y_out_matrix.shape[1])
        if not self.twice_the_same:
            c_bound_k[self.selected_voter_list] = np.inf
        c_bound_k[gamma_h_k <= 0] = np.inf
        c_bound_k[test_alpha >= 0] = np.inf
        c_bound_k[alpha_k < 0] = np.inf

        alpha_k = np.expand_dims(alpha_k, 1)

        c_bound_k_ = y*(F_k+(out*alpha_k.T))
        c_bound_k_ = ((np.sum(c_bound_k_, axis=0)**2.0)
                      /(np.sum(c_bound_k_**2.0, axis=0)))
        c_bound_k_ = 1.0-(1.0/m)*c_bound_k_
        c_bound_k[c_bound_k != np.inf] = c_bound_k_[c_bound_k != np.inf]

        alpha_argmin = np.argmin(c_bound_k)

        # If all C-bound are inf => there is no possible improvement, we stop
        if(c_bound_k[alpha_argmin] == np.inf):
            return False

        # We update the weights of the classifer F_(k+1)
        self.post[alpha_argmin] = self.post[alpha_argmin]+alpha_k[alpha_argmin]
        self.selected_voter_list.append(alpha_argmin)

        return True

    def predict(self, X):
        return self.mv.predict(X)
