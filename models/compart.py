import numpy as np
import time
from covidprocess import DataCovid
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


class SIR:
    """
    The SIR model without vital dynamics. Very simple model uses differential equations:
        dS/dt = -b*I*S/N
        dI/dt = b*I*S -g*I
        dR/dt = g*I
        - where
        S - stock of susceptible population
        I - stock of infected
        R - stock of recovered

        R_0 = b/g - basic reproduction number
        b - infection-producing contacts per unit time = tau
        tau = 1/g -  mean infectious period
        g = 1/tau

        Source: https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
    """
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
        # Init stocks of population
        self._S_0 = data['Susceptible'][0]
        self._I_0 = data['Infected'][0]
        self._R_0 = data['Recovered'][0]
        self._N_0 = data['Population']
        # True values
        self._S_true = data['Susceptible']
        self._I_true = data['Infected']
        self._R_true = data['Recovered']
        self._dt = 0.1

    def _susceptible_new(self, b, S, I):
        N = self._N_0
        dt = self._dt
        return S - b*S*I/N*dt

    def _infected_new(self, b, g, S, I):
        N = self._N_0
        dt = self._dt
        return I + dt*(b*S*I/N - g*I)

    def _recovered_new(self, g, I, R):
        dt = self._dt
        return R + g*I*dt

    def _full_predict(self, x, T=None):
        b, g, S_0, I_0, R_0 = x
        if T is None:
            T = len(self._S_true)

        S = S_0
        I = I_0
        R = R_0
        dt = self._dt

        y_pred = {
            'Susceptible': [],
            'Infected': [],
            'Recovered': []
        }

        for i in range(int(T // dt)):
            S = self._susceptible_new(b, S, I)
            I = self._infected_new(b, g, S, I)
            R = self._recovered_new(g, I, R)

            if i*dt % 1 == 0.0:
                y_pred['Susceptible'].append(S)
                y_pred['Infected'].append(I)
                y_pred['Recovered'].append(R)

        return y_pred

    def _loss(self, x, train_size=None):
        if train_size is None:
            train_size = (0.25, 0.7)
        y_pred = self._full_predict(x)
        # Cut head of prediction and tail of true
        I_pred = y_pred['Infected'][:-1]
        I_true = self._I_true[1:]

        start = int(len(I_true) * train_size[0])
        stop = int(len(I_true) * train_size[1])
        I_pred_train = np.array(I_pred[start:stop])
        I_true_train = np.array(I_true[start:stop])

        sample_weights = [i**2 for i in range(len(I_pred_train))]
        loss = mean_squared_log_error(I_true_train, I_pred_train, sample_weights)
        # loss = mean_squared_error(I_true_train, I_pred_train, sample_weights)
        return loss

    def _optimize(self, info=False, init_args=None, bnds=None):
        if init_args is None:
            init_args = {'b': 3.0, 'g': 3.0,
                         'S_0': self._N_0,
                         'I_0': self._I_0,
                         'R_0': self._R_0
                         }
        if bnds is None:
            bnds = {'b': (-2, None), 'g': (-2, None),
                    'S_0': (0, None),
                    'I_0': (0, None),
                    'R_0': (0, None)
                    }

        opt_parameters = minimize(self._loss, np.array(init_args.values()), bounds=tuple(bnds.values())).x
        loss = self._loss(opt_parameters)
        if info:
            print('Optimal parameters:')
            print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(*opt_parameters))
            print('Loss = {}'.format(loss))

        return opt_parameters, loss

    def _naive_optimize(self, info=False):
        grid = {'b': np.linspace(0, 3, 3),
                'g': np.linspace(0, 3, 3),
                'S_0': np.linspace(0.5*self._N_0, self._N_0, 3),
                'I_0': np.linspace(0, max(self._I_0 * 10, 10), 3),
                'R_0': np.linspace(0, max(self._R_0 * 10, 10), 3)}

        begin_time = time.time()
        loss = 10e9
        opt_init_params = None
        opt_parameters = None
        for b in grid['b']:
            for g in grid['g']:
                for S_0 in grid['S_0']:
                    for I_0 in grid['I_0']:
                        for R_0 in grid['R_0']:
                            init_args = {'b': b, 'g': g,
                                         'S_0': S_0,
                                         'I_0': I_0,
                                         'R_0': R_0
                                         }
                            try:
                                cur_opt_parameters, cur_loss = self._optimize(info, init_args)
                                if info:
                                    print('Current parameters:')
                                    print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(*cur_opt_parameters))
                                    print('Current loss = ', cur_loss)
                                    print('Wait ...')
                            except ValueError:
                                if info:
                                    print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(*init_args))
                                    print(' Bad init parameters. Continue...')
                                    print('Wait ...')
                                continue
                            if cur_loss < loss:
                                opt_parameters = cur_opt_parameters
                                opt_init_params = init_args
                                loss = cur_loss
        end_time = time.time()
        if info:
            print('Final Results')
            print('Best INIT parameters:')
            print('Init b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(*opt_init_params))
            print('Best parameters:')
            print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(*opt_parameters))
            print('Best loss = ', loss)
            print('Time of calculation = {} seconds'.format(round(end_time - begin_time)))

        return opt_parameters, loss

    def fit(self, info=True):
        self._opt_parameters, self._train_loss = self._optimize(info)
        return self

    def predict(self, T_pred=None):
        if T_pred is None:
            T_pred = len(self._I_true)
        try:
            b, g, S_0, I_0, R_0 = self._opt_parameters
        except AttributeError:
            print('Not fitted yet.')
            print('Start fitting. Wait ...')
            self.fit(info=False)
            b, g, S_0, I_0, R_0 = self._opt_parameters

        y_pred = self._full_predict((b, g, S_0, I_0, R_0), T_pred)
        print('Successfully.')
        return y_pred

    def plot(self, T_print=None):
        I_true = self._I_true
        if T_print is None:
            T_print = len(I_true)

        y_pred = self.predict(T_print)
        I_pred = y_pred['Infected']

        true_print_size = min(T_print, len(I_true))
        pred_print_size = max(T_print, len(I_true))
        X_pred_print = list(range(pred_print_size))
        I_pred_print = I_pred[:pred_print_size]

        X_true_print = list(range(true_print_size))

        train_start = int(0.25*len(X_true_print))
        valid_start = int(0.7*len(X_true_print))

        X_true_useless_print = X_true_print[:train_start]
        I_true_useless_print = I_true[:train_start]

        X_true_train_print = X_true_print[train_start:valid_start]
        I_true_train_print = I_true[train_start:valid_start]

        X_true_valid_print = X_true_print[valid_start:]
        I_true_valid_print = I_true[valid_start:]

        plt.figure(figsize=(12, 8))
        plt.title('Infected people')
        plt.xlabel('Days from first case')
        plt.ylabel('Number of infected')
        plt.plot(X_true_useless_print, I_true_useless_print, 'yo--', label='True not used')
        plt.plot(X_true_train_print, I_true_train_print, 'bo--', label='True train')
        plt.plot(X_true_valid_print, I_true_valid_print, 'go--', label='True valid')
        plt.plot(X_pred_print, I_pred_print, 'ro--', label='Prediction')
        plt.legend()
        plt.yscale('log')
        plt.show()


data = DataCovid().read()

model = SIR(data)
model.plot()

