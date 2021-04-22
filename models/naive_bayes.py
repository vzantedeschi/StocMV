import torch

class NaiveBayes():
    """Empirical frequentist or bayesian Naive Bayes classifier in high confidence regime. 
    For reference, see Berend, Daniel, and Aryeh Kontorovich. "A finite sample analysis of the Naive Bayes classifier." J. Mach. Learn. Res. 16.1 (2015): 1519-1545. """

    def __init__(self, voters, frequentist=True):

        super(NaiveBayes, self).__init__()
        self.voters = voters
        self.frequentist = frequentist

    def fit(self, X, y):

        n = len(X)
        y_pred = self.voters(X)

        num_corrects = torch.where(y[:, None] == y_pred, torch.ones(1), torch.zeros(1)).sum(0)

        if self.frequentist:
            p = num_corrects / n
            self.theta = torch.log(p / (1 - p))

        else: # bayesian, with alpha = 1, beta = 1
            self.theta = torch.log((1 + num_corrects)/ (1 + n - num_corrects))

    def predict(self, X):

        y_pred = self.voters(X).transpose(1, 0)

        labels = self.theta @ y_pred

        if y_pred.dim() == 3:
            num_classes = labels.shape[2]
            c_min, c_max = 0, num_classes - 1
            labels = torch.argmax(labels, 2) # if multiclass

        else:
            num_classes = 1
            c_min, c_max = -1, 1
            labels = torch.sign(labels) # if binary

        return labels        