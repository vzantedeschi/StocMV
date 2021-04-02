#!/usr/bin/env python
import importlib
import inspect
import math
import cvxpy as cp

import math

from models.stochastic_mv import MajorityVote
from core.kl_inv import kl_inv


###############################################################################

class MetaMetrics(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.metrics"), inspect.isclass
        ):
            if(class_name != "MetaMetrics" and class_name != "Metrics"):
                #  class_name = class_name.lower()
                class_name = class_name.replace("Metrics", "")
                class_dict[class_name] = class_
        return class_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

        # Getting the name of the module
        if("name" not in kwargs):
            class_name = args[0]
        else:
            class_name = kwargs["name"]

        # Getting the module dictionnary
        class_dict = cls.__get_class_dict()

        # Checking that the module exists
        if(class_name not in class_dict):
            raise Exception(class_name+" doesn't exist")

        # Adding the new module in the base classes
        bases = (class_dict[class_name], )+bases

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaMetrics, new_cls).__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #


class Metrics(metaclass=MetaMetrics):

    def __init__(self, name, majority_vote=None):
        super().__init__()
        self.mv = majority_vote

    def fit(self, y, y_p):
        raise NotImplementedError


# --------------------------------------------------------------------------- #


class RiskMetrics():

    def fit(self, y, y_p):

        return (0.5*(1.0-y_p*y)).mean()

class DisagreementMetrics():

    def fit(self, y, y_p):

        return (0.5*(1.0-y_p**2.0)).mean()


class JointMetrics():

    def fit(self, y, y_p):

        return ((0.5*(1.0-y_p*y))**2.0).mean()


class ZeroOneMetrics():

    def fit(self, y, y_p):

        risk = Metrics("Risk").fit
        y_p_ = 2.0*(y_p > 0.0).float()-1.0
        return risk(y, y_p_)


class CBoundMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):

        m = y.shape[0]

        risk = Metrics("Risk").fit
        disa = Metrics("Disagreement").fit

        rS = risk(y, y_p)
        dS = disa(y, y_p)

        return self.__c_bound(rS, dS)

    def __c_bound(self, r, d):
        r = min(0.5, r)
        d = max(0.0, d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        return cb


class CBoundMcAllesterMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)

        m = y.shape[0]

        risk = Metrics("Risk").fit
        disa = Metrics("Disagreement").fit

        rS = risk(y, y_p)
        dS = disa(y, y_p)

        kl = self.mv.KL()

        r = self.__risk_bound(rS, kl, m)

        if(r > 0.5):
            print(f"Vacuous bound: risk > 0.5")
            return 1.

        d = self.__disagreement_bound(dS, kl, m)
        b = self.__c_bound(r, d)

        print(f"risk={rS}, disagreement={dS}, KL={kl}")
        print(f"Bound={b}, Risk bound={r}, Disagr bound={d}\n")

        return b

    def __risk_bound(self, rS, kl, m):
        b = (1.0/(2.0*m))*(kl+math.log(
            (2.0*math.sqrt(m))/(0.5*self.delta)))
        b = rS + b**0.5
        return b

    def __disagreement_bound(self, dS, kl, m):
        b = (1.0/(2.0*m))*(2.0*kl+math.log(
            (2.0*math.sqrt(m))/(0.5*self.delta)))
        b = dS - b**0.5
        return b

    def __c_bound(self, r, d):
        r = min(0.5, r)
        d = max(0.0, d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        return cb


class CBoundSeegerMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)

        m = y.shape[0]

        risk = Metrics("Risk").fit
        disa = Metrics("Disagreement").fit

        rS = risk(y, y_p)
        dS = disa(y, y_p)

        kl = self.mv.KL()

        r = self.__risk_bound(rS, kl, m)

        if(r > 0.5):
            print(f"Vacuous bound: risk > 0.5")
            return 1.

        d = self.__disagreement_bound(dS, kl, m)
        b = self.__c_bound(r, d)

        print(f"risk={rS}, disagreement={dS}, KL={kl}")
        print(f"Bound={b}, Risk bound={r}, Disagr bound={d}\n")

        return b

    def __bound(self, kl, m):
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(0.5*self.delta)))

    def __risk_bound(self, rS, kl, m):
        b = self.__bound(kl, m)
        b = kl_inv(rS, b, "MAX")
        return b

    def __disagreement_bound(self, dS, kl, m):
        b = self.__bound(2.0*kl, m)
        b = kl_inv(dS, b, "MIN")
        return b

    def __c_bound(self, r, d):
        r = min(0.5, r)
        d = max(0.0, d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        return cb


class CBoundJointMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)

        m = y.shape[0]

        joint = Metrics("Joint").fit
        disa = Metrics("Disagreement").fit

        eS = joint(y, y_p)
        dS = disa(y, y_p)

        kl = self.mv.KL()

        if(2.0*eS+dS >= 1.0 or dS > 2*(eS**0.5 -eS)):
            return 1.

        (e, d) = self.__joint_disagreement_bound(eS, dS, kl, m)
        return self.__c_bound(e, d)

    def __bound(self, kl, m):
        # We compute the PAC-Bayes bound of PAC-Bound 2 (see page 820 of [1])
        b = math.log((2.0*math.sqrt(m)+m)/(self.delta))
        b = (1.0/m)*(2.0*kl+b)
        return b

    def __joint_disagreement_bound(self, eS, dS, kl, m):
        # We solve the inner maximization Problem using the
        # "Bisection method for quasiconvex optimization" of [3] (p 146)
        u = 1.0
        l = 0.0
        bound = self.__bound(kl, m)

        while(u-l > 0.01):
            t = (l+u)/2.0

            e = cp.Variable(shape=1, nonneg=True)
            d = cp.Variable(shape=1, nonneg=True)

            prob = cp.Problem(
                cp.Minimize((1-(2*e+d))**2.0-t*(1-2*d)),
                [(cp.kl_div(eS, e)+cp.kl_div(dS, d)
                  + cp.kl_div((1-eS-dS), 1-e-d) <= bound),
                 2*e+d <= 1,
                 d <= 2.0*(cp.sqrt(e)-e)])

            prob.solve()
            e = e.value[0]
            d = d.value[0]

            c_bound = 1.0-((1-(2*e+d))**2.0)/(1-2*d)

            if(c_bound > 1.0-t):
                u = t
            else:
                l = t
        return (e, d)

    def __c_bound(self, e, d):
        # We compute the C-Bound of PAC-Bound 2 (see page 820 of [1])
        return (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))


class RiskBoundMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)

        m = y.shape[0]

        risk = Metrics("Risk").fit

        rS = risk(y, y_p)

        kl = self.mv.KL()

        r = self.__risk_bound(rS, kl, m)
        return 2.0*r

    def __bound(self, kl, m):
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(self.delta)))

    def __risk_bound(self, rS, kl, m):
        b = self.__bound(kl, m)
        b = kl_inv(rS, b, "MAX")
        return b


class JointBoundMetrics():

    def __init__(self, name, majority_vote=None, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta

    def fit(self, y, y_p):
        assert isinstance(self.mv, MajorityVote)

        m = y.shape[0]

        joint = Metrics("Joint").fit

        eS = joint(y, y_p)

        kl = self.mv.KL()

        e = self.__risk_bound(eS, kl, m)
        return 4.0*e

    def __bound(self, kl, m):
        return (1.0/m)*(kl+math.log((2.0*math.sqrt(m))/(self.delta)))

    def __risk_bound(self, eS, kl, m):
        b = self.__bound(2.0*kl, m)
        b = kl_inv(eS, b, "MAX")
        return b

###############################################################################
