import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar
from scipy import optimize

NumpyArray = TypeVar('numpy.ndarray')


class Forecast(object):
    """ Model to forecast marketshare of a product with time.

    The model is from Nederlands Tijdschrift voor Natuurkunde 83, October 2017,
    page 350-354. This article is based on two previous articles of the
    authors. The one which is not behind a paywall is called
    ‘The cradle of new energy technologies. Why we have solar cells but not yet
    nuclear fusion?,’ by N.L. Cardozo, G. Lange and G.J. Kramer published in
    December 2015 in ‘The colours of energy. Essays on the future of energy and
    society’
    http://www.shell.com/energy-and-innovation/the-energy-future/colours.html#vanity-aHR0cDovL3d3dy5zaGVsbC5jb20vY29sb3Vycw

    amounts : NumpyArray
        The parameter you are trying to forecast.
        Examples: power produced by solar panels or the number of electric
        vehicles in the world.
    years : NumpyArray
        The corresponding year to each data point in *amount*.
    saturation : float
        The saturation value where the growth stops.
        For example, if all cars are replaced by electric vehicles, this value
        is equal to all vehicles currently in the world.
    tau_life : float
        The lifetime of the object in the study, e.g. how often it is replaced.
        Say for cars they are demolished after ~20 years.

    Currently only works with years.
    """

    def __init__(self,
                 amounts: NumpyArray,
                 years: NumpyArray,
                 saturation: float,
                 tau_life: float
                 ):
        self.amounts = amounts
        self.years = years
        self.start_year = years[0]
        self.saturation = saturation
        self.tau_life = tau_life

        # Parameters which are calculated later
        self.tau_exp, self.t_trans = 0, 0

        # Optional plot labels
        self.plot_labels = {
            'xlabel': None,
            'ylabel': None,
            'title': None
        }

    def exp_growth(self, t, tau_exp, t_trans):
        """ Growth during the exponential phase.

        From Nederlands Tijdschrift voor Natuurkunde 83, October 2017,
        page 350-354

        t : int, array
            Time
        tau_exp : float
            Characteristic growth time of the exponential growth
        t_trans : float
            Time at which the exponential growth transitions to linear growth
        """
        return (
            self.saturation * tau_exp / self.tau_life
            * (np.exp((t - t_trans) / tau_exp)
            - np.exp((t - t_trans - self.tau_life) / tau_exp))
        )

    def lin_growth(self, t):
        """ Growth during the linear phase.

        From Nederlands Tijdschrift voor Natuurkunde 83, October 2017,
        page 350-354

        t : int, array
            Time
        """
        # Check if required values are present
        if any([self.tau_exp == 0, self.t_trans == 0]):
            self.fit_exp_phase()

        return (
            self.saturation * self.tau_exp / self.tau_life
            * (1 + (t - self.t_trans) / self.tau_exp
            - np.exp((t - self.t_trans - self.tau_life) / self.tau_exp))
        )

    def fit_exp_phase(self):
        """ Fit the exponential growth phase. """
        popt, _ = optimize.curve_fit(
            lambda t, tau, t_trans: np.log(self.exp_growth(t, tau, t_trans)),
            self.years - self.start_year,
            np.log(self.amounts)
        )
        self.tau_exp, self.t_trans = popt

    def get_transistion_year(self):
        """ Returns the year at which the growth changes from exp to lin. """
        self.fit_exp_phase()

        return int(np.around(self.t_trans) + self.start_year)

    def get_saturation_year(self):
        """ Returns the year at which the growth saturates (stops). """

        return self.get_transistion_year() + self.tau_life

    def set_plot_labels(self, xlabel, ylabel, title):
        """ Set the labels for the plot. """
        self.plot_labels['xlabel'] = xlabel
        self.plot_labels['ylabel'] = ylabel
        self.plot_labels['title'] = title

    def add_labels_to_plot(self, ax, labels):
        """ Adds labels to your plot.

        ax : str
            The axis you want to add your labels to. For example: plt or ax.
        labels : list of str
            Labels you want to add to your plot.
        """

        if 'xlabel' in labels:
            try:
                ax.xlabel(self.plot_labels['xlabel'])
            except AttributeError:
                ax.set_xlabel(self.plot_labels['xlabel'])
        if 'ylabel' in labels:
            try:
                ax.ylabel(self.plot_labels['ylabel'])
            except AttributeError:
                ax.set_ylabel(self.plot_labels['ylabel'])
        if 'title' in labels:
            try:
                ax.title(self.plot_labels['title'])
            except (TypeError, AttributeError):
                ax.set_title(self.plot_labels['title'])

    def plot_exponential_phase(self, figsize=(8, 6)):
        """ Scatter plot of the data with the exponential fit as a line.

        Use this to check how good the fit is.
        """

        self.fit_exp_phase()

        plt.figure(figsize=figsize)
        plt.scatter(self.years, self.amounts)
        plt.plot(
            self.years,
            self.exp_growth(
                self.years - self.start_year,
                self.tau_exp,
                self.t_trans
            )
        )

        labels = [
            name for name, label in self.plot_labels.items()
            if label is not None
        ]
        self.add_labels_to_plot(plt, labels)
        plt.yscale('log')
        plt.show()

    def plot_all_phases(self, figsize=(8, 6)):
        """ Shows all the different growth phases on a log and lin scale. """
        self.fit_exp_phase()

        t_trans_round = int(np.around(self.t_trans)) + self.start_year

        exp_years = np.array(
            [*range(self.start_year, t_trans_round + 1)]
        )
        lin_years = np.array(
            [*range(t_trans_round, t_trans_round + self.tau_life)]
        )

        f, axarr = plt.subplots(2, sharex=True, figsize=figsize)
        for i, ax in enumerate(axarr):
            ax.scatter(self.years, self.amounts)
            # Exponential growth phase
            exp_amounts = self.exp_growth(
                    exp_years - self.start_year,
                    self.tau_exp,
                    self.t_trans
            )
            ax.plot(exp_years, exp_amounts, zorder=1)

            # Linear growth phase
            lin_amounts = self.lin_growth(lin_years - self.start_year)
            ax.plot(lin_years, lin_amounts, zorder=1)

            # Saturation phase
            sat_years = [*range(lin_years[-1], lin_years[-1] + 5)]
            ax.plot(
                sat_years,
                [lin_amounts[-1]] + [self.saturation] * (len(sat_years) - 1),
                zorder=1
            )

            # Place a white dot between phases to visually separate them
            ax.scatter(
                [exp_years[-1], lin_years[-1]],
                [exp_amounts[-1], lin_amounts[-1]],
                zorder=2,
                color='w'
            )

            if i == 0:
                ax.set_yscale('log')
                self.add_labels_to_plot(ax, ['ylabel', 'title'])
            else:
                self.add_labels_to_plot(ax, ['xlabel', 'ylabel'])
        plt.show()

if __name__ == "__main__":
    # Worldwide battery electric vehicles stock
    # Source: Table 5, https://www.iea.org/publications/freepublications/publication/GlobalEVOutlook2017.pdf
    EV_stock = np.array([
        1370.0,
        1690.0,
        2150.0,
        4540.0,
        7470.0,
        16420.0,
        55160.0,
        112940.0,
        226780.0,
        420330.0,
        745610.0,
        1208900.0
    ])
    stock_years = np.array([year for year, _ in enumerate(EV_stock, 2005)])

    """
    *tau_life* is the lifetime of a car. Currently (2017) in the Netherlands cars
    are on average 18 years old when they are demolished. So we'll set
    *tau_life*=20 years.
    Source: https://www.cbs.nl/nl-nl/nieuws/2016/20/personenauto-s-steeds-ouder

    The *saturation* we'll set to the amount of vehicles in use in the world in
    2014, because I assume all vehicles will be replaced.
    Source: https://en.wikipedia.org/wiki/Motor_vehicle#cite_note-USeDataBook30-11

    """
    ev_forecast = Forecast(EV_stock, stock_years, saturation=1.2e9, tau_life=20)
    ev_forecast.set_plot_labels(
        'Year', 'Stock',
        'Worldwide stock of battery electric vehicles'
    )

    print(ev_forecast.get_transistion_year(), ev_forecast.get_saturation_year())

    ev_forecast.plot_exponential_phase(figsize=(5, 5))
    ev_forecast.plot_all_phases(figsize=(15, 8))



