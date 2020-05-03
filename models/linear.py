import numpy as np
from abc import ABC, abstractmethod

from covidprocess import DataCovid

from matplotlib import pyplot as plt
import seaborn as sns
sns.set()



class LinearFitter:

    def __init__(self, X, Y, loss='mse'):
        assert loss == 'mse', 'Loss does not exists'
        self._loss = loss
        self._X = X
        self._Y = Y

    def fit(self, learning_rate=0.0005, num_iterations=1_000_000):
        def mse(y_true, y_pred):
            """ Calculates mean squared error.
                y_true - np.array
                y_pred - np.array
            """
            m = len(y_true)
            loss = 1 / m * np.sum(np.power((y_true - y_pred), 2))
            return loss

        def init_weights(n_x):
            w = np.random.randn(n_x, 1)
            b = 0
            return w, b

        def compute_grads(X, A, Y):
            """ X = np.array, X.shape = (n_x, m) - features
                Y = np.array, Y.shape = (1, m) - true
                A = np.array, A.shape = (1, m) - prediction

                A = dot(w.T, X) + b
                dw = dL/da * da/dw = 1/m * dot(X, (A - Y).T)
                db = dL/da * da/db = 1/m * sum(A - Y)
                """
            m = X.shape[1]
            dw = 1 / m * np.dot(X, (A - Y).T)
            db = 1 / m * np.sum(A - Y)
            grads = {'dw': dw,
                     'db': db}
            return grads

        def update_weights(w, b, grads, learning_rate):
            dw = grads['dw']
            db = grads['db']
            w -= dw * learning_rate
            b -= db * learning_rate
            return w, b

        def train_model(X, Y, learning_rate, num_iterations):
            n_x = X.shape[0]
            w, b = init_weights(n_x)
            for _ in range(num_iterations):
                A = np.dot(w.T, X) + b
                grads = compute_grads(X, A, Y)
                w, b = update_weights(w, b, grads, learning_rate)
            loss = mse(Y, A)
            print('loss =', loss)
            return w, b

        X, Y = self._X, self._Y
        w, b = train_model(X, Y, learning_rate, num_iterations)
        self._w, self._b = w, b
        return w, b


class LinearModel(ABC):
    """ Base class for liniar models."""
    def __init__(self, data):
        """
        Arguments:
            data - dictionary like input dataset from module 'covidprocess'.
            data = {
                'Susceptible': np.array,
                'Infected': np.array,
                'Recovered': np.array,
                'Dead': np.array,
                'Population': int,
                'First Date': str,
                'Country': str
        }
        """
        self._data = data

    @abstractmethod
    def fit(self, train_window, valid_window, info):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def plot(self):
        pass


class NaiveLinearModel(LinearModel):

    def _train_valid_split(self, start, train_window, valid_window):
        """
        Generates batch of train, test, valid data.
        Arguments:
            start - first index in full data : int
            train_window - train size : int
            valid_window - validation size : int

        """

        data = self._data
        stop_train = min(start + train_window, len(data['Infected']))
        stop_valid = min(start + train_window + valid_window, len(data['Infected']))
        stop_test = len(data['Infected'])

        X_train = np.array([i for i in range(start, stop_train)]).reshape(1, -1)
        y_train = {'Infected': data['Infected'][start:stop_train],
                   'Dead': data['Dead'][start:stop_train]}
        X_valid = np.array([i for i in range(stop_train, stop_valid)]).reshape(1, -1)
        y_valid = {'Infected': data['Infected'][stop_train:stop_valid],
                   'Dead': data['Dead'][stop_train:stop_valid]}
        X_test = np.array([i for i in range(stop_valid, stop_test)]).reshape(1, -1)
        y_test = {'Infected': data['Infected'][stop_valid:stop_test],
                  'Dead': data['Dead'][stop_valid:stop_test]}

        for sample in [y_train, y_valid, y_test]:
            for group in sample:
                sample[group] = np.array(sample[group]).reshape(1, -1)

        return y_train, X_train, y_valid, X_valid, y_test, X_test

    def fit(self, start=0, train_window=14, valid_window=7, info=True):
        y_train, X_train, y_valid, X_valid, y_test, X_test = self._train_valid_split(start, train_window, valid_window)
        w = {'Infected': None, 'Dead': None}
        b = {'Infected': None, 'Dead': None}
        for group in y_train:
            y_train_log = np.log(1.0001 + y_train[group])
            fitter = LinearFitter(X_train, y_train_log, 'mse')
            w[group], b[group] = fitter.fit()

        self._X = {'train': X_train, 'valid': X_valid, 'test': X_test}
        self._Y = {'train': y_train, 'valid': y_valid, 'test': y_test}
        self._w = w
        self._b = b
        if info:
            print('Fitted succesfully')
        return self

    def predict(self, group='All', info=True, start_fit=0):
        assert group in ['All', 'Dead', 'Infected'], 'Group does not exists'
        w, b = self._w, self._b
        if group == 'All':
            y_pred = {'Infected': None, 'Dead': None}
        else:
            y_pred = {group: None}

        for group in y_pred:
            w, b = self._w[group], self._b[group]
            X = self._X
            X_train, X_valid, X_test = X['train'], X['valid'], X['test']
            y_pred[group] = {'train': np.exp(np.dot(w.T, X_train) + b),
                             'valid': np.exp(np.dot(w.T, X_valid) + b),
                             'test': np.exp((np.dot(w.T, X_test)) + b)}
        self._prediction = y_pred
        if info:
            print('Predicted succesfully')
        return y_pred

    def plot(self, start_fit=0, info=True, group='All'):
        try:
            prediction = self._prediction
        except AttributeError:
            print('Not predicted yet.\nWait for prediction ...')
            try:
                prediction = self.predict(group)
            except AttributeError:
                print('Not fitted yet.\nWait for fitting ...')
                self.fit(start_fit)
                prediction = self.predict(group)

        X = self._X
        Y = self._Y
        X_train, X_valid, X_test = X['train'], X['valid'], X['test']
        Y_train, Y_valid, Y_test = Y['train'], Y['valid'], Y['test']
        X_all = np.concatenate((X_train, X_valid, X_test), axis=1)
        for group in prediction:
            Y_all = np.concatenate((Y_train[group], Y_valid[group], Y_test[group]), axis=1)
            plt.figure(figsize=(12, 8))
            plt.title(group + ' people')
            plt.xlabel('Days from first case')
            plt.ylabel('Number of ' + group)
            plt.plot(X_train[0], prediction[group]['train'][0], 'go--', label='Prediction of ' + group + ' train')
            plt.plot(X_valid[0], prediction[group]['valid'][0], 'bo--', label='Prediction of ' + group + ' valid')
            plt.plot(X_test[0], prediction[group]['test'][0], 'yo--', label='Prediction of ' + group + ' test')
            plt.plot(X_all[0], Y_all[0], 'ro--', label='True values')
            plt.legend()
            plt.yscale('log')
        plt.show()


data = DataCovid().read()
linear_model = NaiveLinearModel(data)
linear_model.plot(start_fit=40)