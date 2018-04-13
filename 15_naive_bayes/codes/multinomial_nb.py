import numpy as np

class MultinomialNB(object):
    def __init__(self, alpha, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.classes = None
        self.class_prior = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        # class prior probabilities: P(y=ck)
        self.cal_prior_prob(y)
        # conditional probabilities: P(xj|y=ck)
        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c] = {}
            for i in xrange(len(X[0])): # 对于每个特征
                sample_f = X[np.equal(y, c)][:, i]
                self.conditional_prob[c][i] = self.cal_conditional_prob(sample_f)
        return self

    def cal_prior_prob(self, y):
        class_num = len(self.classes)
        if self.fit_prior:
            self.class_prior = [1 / class_num for _ in xrange(class_num)]
        else:
            self.class_prior = []
            sample_num = float(len(y))
            for c in self.classes:
                c_num = np.sum(np.equal(y, c))
                self.class_prior.append(c_num + self.alpha / (sample_num + self.alpha * class_num))

    def cal_conditional_prob(self, sample_f):
        values = np.unique(sample_f)
        total_num, value_num = len(sample_f), len(values)
        value_prob = {}
        for v in values:
            value_prob[v] = ((np.sum(np.equal(sample_f, v)) + self.alpha) /
                             (total_num + self.alpha * value_num))
        return value_prob

    def predict(self, X):
        labels = []
        if X.ndim == 1:
            labels.append(self.predict_one_sample(X))
        else:
            for i in xrange(X.shape[0]):
                labels.append(self.predict_one_sample(X[i]))
        return labels

    def predict_one_sample(self, X):
        # for each category, calculate its posterior probability: class_prior * conditional_prob
        label, max_prob = -1, 0.0
        for c_index in xrange(len(self.classes)):
            current_class_prob = self.classes[c_index]
            current_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c_index]]
            j = 0
            for i in feature_prob.keys():
                current_conditional_prob *= self.predict_ij(feature_prob[i], j)
            if current_class_prob * current_conditional_prob > max_prob:
                max_prob = current_class_prob * current_conditional_prob
                label = self.classes[c_index]
        return label

    def predict_ij(self, value_prob, j):
        return value_prob[j]