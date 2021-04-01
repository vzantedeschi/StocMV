#!/usr/bin/env python
import argparse
import logging
import numpy as np
from h5py import File

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from learner.mincq_learner import MinCqLearner
from learner.mincq_learner_v2 import MinCqLearnerV2

from voter.stump import DecisionStumpMV
from voter.tree import TreeMV

from core.metrics import Metrics
from core.save import save_csv

from sklearn.tree import export_text

###############################################################################


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().disabled = True

    # ----------------------------------------------------------------------- #

    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="path csv")
    arg_parser.add_argument(
        "name", metavar="name", type=str,
        help="name experiment")

    arg_parser.add_argument(
        "--voter", metavar="voter", default="stump", type=str,
        help="voter")
    arg_parser.add_argument(
        "--nb-per-attribute", metavar="nb-per-attribute", default=10, type=int,
        help="nb-tree")
    arg_parser.add_argument(
        "--nb-tree", metavar="nb-tree", default=100, type=int,
        help="nb-tree")
    arg_parser.add_argument(
        "--prior", metavar="prior", default=0.5, type=float,
        help="prior")
    arg_parser.add_argument(
        "--not-quasi-uniform", default=True, action="store_false",
        help="not-quasi-uniform")

    arg_list = arg_parser.parse_args()
    path = arg_list.path
    name = arg_list.name
    voter = arg_list.voter
    nb_per_attribute = arg_list.nb_per_attribute
    nb_tree = arg_list.nb_tree
    prior = arg_list.prior
    quasi_uniform = arg_list.not_quasi_uniform

    NB_PARAMS = 20

    # ----------------------------------------------------------------------- #

    data = File("data/"+arg_list.data+".h5", "r")

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    y_train = np.expand_dims(y_train, 1)
    x_test = np.array(data["x_test"])
    y_test = np.array(data["y_test"])
    y_test = np.expand_dims(y_test, 1)

    assert len(x_train.shape) == 2 and len(x_test.shape) == 2
    assert len(y_train.shape) == 2 and len(y_test.shape) == 2
    assert x_train.shape[0] == y_train.shape[0] and x_train.shape[0] > 0
    assert x_test.shape[0] == y_test.shape[0] and x_train.shape[0] > 0
    assert y_train.shape[1] == y_test.shape[1] and y_train.shape[1] == 1
    y_unique = np.sort(np.unique(y_train))
    assert y_unique[0] == -1 and y_unique[1] == +1
    y_unique = np.sort(np.unique(y_test))
    assert y_unique[0] == -1 and y_unique[1] == +1

    assert voter == "stump" or voter == "tree"
    if(voter == "stump"):
        VOTER = 0
        majority_vote = DecisionStumpMV(
            x_train, y_train,
            nb_per_attribute=nb_per_attribute,
            complemented=True, quasi_uniform=quasi_uniform)
    elif(voter == "tree"):
        VOTER = 1
        x_prior = x_train[:int(prior*len(x_train)), :]
        y_prior = y_train[:int(prior*len(y_train)), :]
        x_train = x_train[int(prior*len(x_train)):, :]
        y_train = y_train[int(prior*len(y_train)):, :]
        majority_vote = TreeMV(
            x_prior, y_prior,
            nb_tree=nb_tree,
            complemented=True, quasi_uniform=quasi_uniform)

    if(quasi_uniform):
        learner = MinCqLearner(majority_vote, 0.1)
    else:
        learner = MinCqLearnerV2(majority_vote, 0.1)

    learner_params = {'mu': np.linspace(10**(-4), 0.5, NB_PARAMS)}
    zero_one = Metrics("ZeroOne").fit

    cv_classifier = GridSearchCV(
        learner, learner_params, cv=3,
        scoring=make_scorer(zero_one, greater_is_better=False), refit=False)
    cv_classifier = cv_classifier.fit(x_train, y_train)

    mu_best = cv_classifier.best_params_["mu"]

    if(quasi_uniform):
        learner = MinCqLearner(majority_vote, mu_best)
    else:
        learner = MinCqLearnerV2(majority_vote, mu_best)
    learner = learner.fit(x_train, y_train)

    # ----------------------------------------------------------------------- #

    risk = Metrics("Risk").fit
    disa = Metrics("Disagreement").fit
    joint = Metrics("Joint").fit
    zero_one = Metrics("ZeroOne").fit

    c_bound = Metrics("CBound").fit
    c_bound_mcallester = Metrics(
        "CBoundMcAllester", majority_vote, delta=0.05/float(NB_PARAMS)).fit
    c_bound_seeger = Metrics(
        "CBoundSeeger", majority_vote, delta=0.05/float(NB_PARAMS)).fit
    c_bound_joint = Metrics(
        "CBoundJoint", majority_vote, delta=0.05/float(NB_PARAMS)).fit
    risk_bound = Metrics(
        "RiskBound", majority_vote, delta=0.05/float(NB_PARAMS)).fit
    joint_bound = Metrics(
        "JointBound", majority_vote, delta=0.05/float(NB_PARAMS)).fit

    y_p_train = learner.predict(x_train)
    y_p_test = learner.predict(x_test)

    zero_one_S = zero_one(y_train, y_p_train)
    rS = risk(y_train, y_p_train)
    dS = disa(y_train, y_p_train)
    eS = joint(y_train, y_p_train)

    zero_one_T = zero_one(y_test, y_p_test)
    rT = risk(y_test, y_p_test)
    dT = disa(y_test, y_p_test)
    eT = joint(y_test, y_p_test)

    cb_mc = c_bound_mcallester(y_train, y_p_train)
    cb_se = c_bound_seeger(y_train, y_p_train)
    cb_jo = c_bound_joint(y_train, y_p_train)
    risk_b = risk_bound(y_train, y_p_train)
    joint_b = joint_bound(y_train, y_p_train)

    cb_S = c_bound(y_train, y_p_train)
    cb_T = c_bound(y_test, y_p_test)

    save_csv(path, {
        "zero_one_S": zero_one_S,
        "rS": rS,
        "dS": dS,
        "eS": eS,
        "zero_one_T": zero_one_T,
        "rT": rT,
        "dT": dT,
        "eT": eT,
        "c_bound_S": cb_S,
        "c_bound_T": cb_T,
        "c_bound_mcallester": cb_mc,
        "c_bound_seeger": cb_se,
        "c_bound_joint": cb_jo,
        "risk_bound": risk_b,
        "joint_bound": joint_b,
        "mu": mu_best,
        "voter": VOTER,
        "nb_per_attribute": nb_per_attribute,
        "nb_tree": nb_tree,
        "prior": prior,
    }, name, erase=True)


###############################################################################
