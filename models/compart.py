import numpy as np
import time

from sklearn.metrics import mean_squared_log_error, mean_squared_error
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import seaborn as sns

from .covidprocess import DataCovid

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
            data : dictionary like input dataset from module 'covidprocess'.
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
        # Time integration step, days
        self._dt = 0.1

    def _susceptible_new(self, b, S, I):
        """ Number of susceptible in one time integration step."""
        N = self._N_0
        dt = self._dt
        return S - b * S * I / N * dt

    def _infected_new(self, b, g, S, I):
        """ Number of infected in one time integration step."""
        N = self._N_0
        dt = self._dt
        return I + dt * (b * S * I / N - g * I)

    def _recovered_new(self, g, I, R):
        """ Number of recovered in one time integration step."""
        dt = self._dt
        return R + g * I * dt

    def _full_predict(self, x, T=None):
        """
        Predict cases for each day in range [0..T] days.
        Arguments:
            x : array like with parameters b, g, S_0, I_0, R_0
            T : int, range for prediction
        Returns:
            y_pred = {'Susceptible': array, 'Infected': array, 'Recovered': array}

        """
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

            if i * dt % 1 == 0.0:
                y_pred['Susceptible'].append(S)
                y_pred['Infected'].append(I)
                y_pred['Recovered'].append(R)

        return y_pred

    def _loss(self, x, train_size=None, error='msle'):
        """
        Compute loss of model.
        Arguments:
            x : array like with parameters b, g, S_0, I_0, R_0
            train_size : tuple like (0.25, 0.75),
                where first value - start, second - stop
            error : string, type of loss

        """
        if train_size is None:
            train_size = (0.25, 0.75)

        assert error in ['mse', 'msle'], 'Error not found'

        # Make prediction
        y_pred = self._full_predict(x)

        # Cut head of prediction and tail of true
        I_pred = y_pred['Infected'][:-1]
        I_true = self._I_true[1:]

        # Choose train samples
        start = int(len(I_true) * train_size[0])
        stop = int(len(I_true) * train_size[1])
        I_pred_train = np.array(I_pred[start:stop])
        I_true_train = np.array(I_true[start:stop])

        # Init weight of each point, earlier point - less weight
        sample_weights = [i ** 2 for i in range(len(I_pred_train))]

        # Compute loss
        if error == 'msle':
            loss = mean_squared_log_error(I_true_train, I_pred_train, sample_weights)
        elif error == 'mse':
            loss = mean_squared_error(I_true_train, I_pred_train, sample_weights)
        else:
            raise ValueError('Error not found.')
        return loss

    def _optimize(self, info=False, init_args=None, bounds=None, error='msle'):
        """
        Optimize model by minimizing loss on true values.
        Arguments:
            info: bool, print information in console if True.
            init_args: dictionary with init parameters of b, g, S_0, I_0, R_0 for optimizing.
            bounds: dictionary with bounds of parameters.
            error: string, type of loss

        """
        if init_args is None:
            init_args = {'b': 3.0,
                         'g': 3.0,
                         'S_0': self._N_0,
                         'I_0': self._I_0,
                         'R_0': self._R_0}
        if bounds is None:
            bounds = {'b': (-2, None),
                      'g': (-2, None),
                      'S_0': (0, None),
                      'I_0': (0, None),
                      'R_0': (0, None)}

        init_args_values = np.array([init_args[key] for key in['b', 'g', 'S_0', 'I_0', 'R_0']])
        bounds_values = tuple([bounds[key] for key in['b', 'g', 'S_0', 'I_0', 'R_0']])
        opt_parameters = minimize(self._loss, init_args_values, bounds=bounds_values).x
        loss = self._loss(x=opt_parameters, error=error)
        if info:
            print('Optimal parameters:')
            print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(*opt_parameters))
            print('Loss = {}'.format(loss))

        return opt_parameters, loss

    def _naive_optimize(self, info=False):
        """ Naive optimizer on linear grid of parameters."""
        grid = {'b': np.linspace(0, 3, 3),
                'g': np.linspace(0, 3, 3),
                'S_0': np.linspace(0.5 * self._N_0, self._N_0, 3),
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
                            init_args = {'b': b,
                                         'g': g,
                                         'S_0': S_0,
                                         'I_0': I_0,
                                         'R_0': R_0}
                            try:
                                cur_opt_parameters, cur_loss = self._optimize(info, init_args)
                                if info:
                                    print('Current parameters:')
                                    print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(
                                        *cur_opt_parameters))
                                    print('Current loss = ', cur_loss)
                                    print('Wait ...')
                            except ValueError:
                                if info:
                                    print('b = {}, g = {},\n S_0 = {}, I_0 = {}, R_0 = {}'.format(
                                        *init_args))
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
        """ Fitting on self data."""
        self._opt_parameters, self._train_loss = self._optimize(info)
        return self

    def predict(self, T_pred=None):
        """ Predict values for each day in [0..T_pred]. """
        # Init default value of T_pred
        if T_pred is None:
            T_pred = len(self._I_true)

        # Predict even if not fitted yet
        try:
            b, g, S_0, I_0, R_0 = self._opt_parameters
        except AttributeError:
            print('Not fitted yet.\nStart fitting. Wait ...')
            self.fit(info=False)
            b, g, S_0, I_0, R_0 = self._opt_parameters

        y_pred = self._full_predict(x=(b, g, S_0, I_0, R_0), T=T_pred)
        print('Successfully.')
        return y_pred

    def plot(self, T_plot=None, group='All'):
        """
        Plot values of aim group for each day in [0..T_plot].
        Arguments:
            T_plot: int, range for plot
            group: str, group for plot from ['Infected', 'Recovered', 'All']
        """
        assert group in ['Infected', 'Recovered', 'All'], 'Group not found.'
        # Init default value of T_plot
        if T_plot is None:
            T_plot = len(self._I_true)

        # Choose groups for plotting
        if group == 'All':
            aim_groups = ['Infected', 'Recovered']
        else:
            aim_groups = [group]

        # Make prediction
        y_pred = self.predict(T_plot)

        # Plotting for each aim group
        for group in aim_groups:
            if group == 'Infected':
                Y_true = self._I_true
            else:
                Y_true = self._R_true

            Y_pred = y_pred[group]

            # Make samples for plot
            true_plot_size = min(T_plot, len(Y_true))
            pred_plot_size = max(T_plot, len(Y_true))
            X_pred_plot = list(range(pred_plot_size))
            Y_pred_plot = Y_pred[:pred_plot_size]

            X_true_plot = list(range(true_plot_size))

            train_start = int(0.25 * len(X_true_plot))
            valid_start = int(0.75 * len(X_true_plot))

            X_true_useless_plot = X_true_plot[:train_start]
            Y_true_useless_plot = Y_true[:train_start]

            X_true_train_plot = X_true_plot[train_start:valid_start]
            Y_true_train_plot = Y_true[train_start:valid_start]

            X_true_valid_plot = X_true_plot[valid_start:]
            Y_true_valid_plot = Y_true[valid_start:]

            # Plotting
            plt.figure(figsize=(12, 8))
            plt.title(group + ' people')
            plt.xlabel('Days from first case')
            plt.ylabel('Number of ' + group)
            plt.plot(X_true_useless_plot, Y_true_useless_plot, 'yo--', label='True not used')
            plt.plot(X_true_train_plot, Y_true_train_plot, 'bo--', label='True train')
            plt.plot(X_true_valid_plot, Y_true_valid_plot, 'go--', label='True valid')
            plt.plot(X_pred_plot, Y_pred_plot, 'ro--', label='Prediction')
            plt.legend()
            plt.yscale('log')
        plt.show()

