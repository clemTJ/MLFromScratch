import numpy as np
from collections import Counter

import matplotlib.pyplot as plt


class Model:

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    @staticmethod
    def _most_common(values: list):
        return Counter(values).most_common(1)[0][0]

    def summary(self, ):
        pass

    @staticmethod
    def confusion_matrix(y_true, y_pred, normalize: bool = True):
        labels = sorted(np.unique(y_true))
        cm = []

        for true_label in labels:
            label_cm = []
            true = np.count_nonzero(y_true == true_label)

            for pred_label in labels:
                pred = np.count_nonzero((y_pred == pred_label) & (y_true == true_label))
                if normalize:
                    pred /= true
                label_cm.append(pred)

            cm.append(label_cm)

        cm = np.array(cm)
        return cm

    def summary(self, y_true, y_pred):
        print(f'Accuracy = {self.accuracy(y_true, y_pred)}')
        print(f'MSE = {self.mse(y_true, y_pred)}')
        print(f'MAE = {self.mae(y_true, y_pred)}')
        print(f'R2 = {self.r2(y_true, y_pred)}')
        self.display_confusion_matrix(y_true, y_pred)

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def display_confusion_matrix(y_true, y_pred, normalize: bool = True):
        labels = sorted(np.unique(y_true))
        cm = Model.confusion_matrix(y_true, y_pred, normalize)

        txt_labels = [str(label) for label in labels]
        fig, ax = plt.subplots()

        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')

        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(txt_labels)
        ax.set_yticklabels(txt_labels)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{round(val * 100, 2)}%', ha='center', va='center', color='red')

        plt.show()

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def r2(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
