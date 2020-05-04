import pandas as pd
import numpy as np

DEFAULT_PATHS = {
    'PATH_COVID': r"E:\Practice\Challenges\ODS\covid19\Compartmental models\csse_covid_19_time_series",
    'CONFIRMED_FILENAME': r"\time_series_covid19_confirmed_global.csv",
    'DEATHS_FILENAME': r"\time_series_covid19_deaths_global.csv",
    'RECOVERED_FILENAME': r"\time_series_covid19_recovered_global.csv",
    'PATH_COUNTRIES': r"E:\Practice\Challenges\ODS\covid19\Compartmental models\countries",
    'COUNTRIES_FILENAME': r"\countries.csv"
}


class DataCovid:
    """ Class of simple data extraction from csv files."""
    def __init__(self, country_name='Russia', country_code='RUS'):
        """
        Sometimes there may be no matches.
        Try to find the country in the file manually in case of unsuccessful search.

        Arguments:
            country_name - utf-8 string
            country_code - utf-8 string, Alpha-3 code.

        """
        self._country_name = country_name
        self._country_code = country_code.upper()
        assert (len(country_code) == 3) or (), 'Not Alpha-3 code'

    def read(self, default_paths=None):
        """
        Read and extract data only for one country.

        Arguments:
            default_paths - dictionary with paths to data

        """
        if default_paths is None:
            default_paths = DEFAULT_PATHS

        country_name = self._country_name
        country_code = self._country_code
        # Init paths to data
        conf_file = default_paths['PATH_COVID'] + default_paths['CONFIRMED_FILENAME']
        dead_file = default_paths['PATH_COVID'] + default_paths['DEATHS_FILENAME']
        recov_file = default_paths['PATH_COVID'] + default_paths['RECOVERED_FILENAME']
        countries_file = default_paths['PATH_COUNTRIES'] + default_paths['COUNTRIES_FILENAME']

        # Read raw data with all countries
        confirmed_data = pd.read_csv(conf_file)
        dead_data = pd.read_csv(dead_file)
        recovered_data = pd.read_csv(recov_file)
        countries_data = pd.read_csv(countries_file)

        # Check if country not in data
        assert country_name in confirmed_data['Country/Region'].values, 'Country not found'
        assert country_name in dead_data['Country/Region'].values, 'Country not found'
        assert country_name in recovered_data['Country/Region'].values, 'Country not found'
        assert country_code in countries_data['iso_alpha3'].values, 'Country not found'

        # Extract only one country
        confirmed = confirmed_data[confirmed_data['Country/Region'] == country_name]
        recovered = recovered_data[recovered_data['Country/Region'] == country_name]
        dead = dead_data[dead_data['Country/Region'] == country_name]
        country = countries_data[countries_data['iso_alpha3'] == country_code]
        population = int(country['population'].values)

        # Find first case of infection in time series data
        first_case_date = ''
        for column in confirmed.columns[5:]:
            if confirmed[column].values[0] > 0:
                first_case_date = str(column)
                break
        assert first_case_date != '', 'First case of infection not found'

        confirmed = confirmed.loc[:, first_case_date:].values[0]
        recovered = recovered.loc[:, first_case_date:].values[0]
        dead = dead.loc[:, first_case_date:].values[0]
        susceptible = np.array([(population - confirmed_cases) for confirmed_cases in confirmed])

        data = {
            'Susceptible': susceptible,
            'Infected': confirmed,
            'Recovered': recovered,
            'Dead': dead,
            'Population': population,
            'First Date': first_case_date,
            'Country': country_name
        }
        self.data = data
        return data