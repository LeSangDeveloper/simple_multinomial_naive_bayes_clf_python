import numpy as np

class MultinomialNB():

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, X):
        results = dict()
        for label, values in self.params.items():
            indices = np.where(X != 0)[1]
            results[label] = np.power(values[indices[0]], X[0][indices[0]]) * self.probability_class[label]
            for index in indices[1:]:
                results[label] = results[label] * np.power(values[index], X[0][index])
        total = 0
        for _, value in results.items():
            total += value
        for key, value in results.items():
            results[key] = value / total
        return results

    def fit(self, X, y):
        classes = np.unique(y)
        self.cls_number = {}
        
        i = 0
        for class_name in classes:
            self.cls_number[class_name] = i
            i = i + 1
        
        self.probability_class = {}
        for class_name, number in self.cls_number.items():
            count_temp = 0
            for label in y:
                if label == class_name:
                    count_temp = count_temp + 1
            self.probability_class[class_name] = count_temp / len(y)

        self.total_count = dict()
        for feature, label in zip(X, y):
            if label not in self.total_count.keys():
                self.total_count[label] = feature
            else:
                self.total_count[label] = self.total_count[label] + feature
        
        self.N_class = dict()
        for label, values in self.total_count.items():
            self.N_class[label] = values.sum()

        self.features_length = len(X[0])

        self.params = {}
        for label, values in self.total_count.items():
            self.params[label] = (values + self.alpha) / (self.N_class[label] +  self.features_length * self.alpha)

        return self

        
