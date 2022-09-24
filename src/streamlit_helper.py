# helper functions for streamlit app

from defs import EVENT_DATE_COL, EVENT_NAME_COL, FIRST_SEASON, ABBREVIATION_COL
from utils import get_current_year
from laps_plots import PLOT_FEATURES, DRIVER_NAMES, DRIVER_NAME, REF_DRIVER_NAMES, SHOW_PERSONAL_BEST, SHOW_PIT_STOP, \
    SHOW_TRACK_STATUS

import fastf1 as ff1
import pandas as pd
import streamlit as st


def get_seasons():
    """
    get list of all available seasons
    :return: list of integers
    """
    # get current year
    current_year = get_current_year()

    # check if there are events of the current year
    if not ff1.get_event_schedule(current_year, include_testing=False).empty:
        last_season = current_year
    else:
        last_season = current_year - 1

    return list(range(FIRST_SEASON, last_season + 1))


def get_race_event_names(season):
    """
    get list of all races of selected year
    :param season: integer of year
    :return: list of event names
    """
    # get events of selected year
    event_df = ff1.get_event_schedule(season, include_testing=False)

    # get current date
    current_date = pd.to_datetime(pd.Timestamp.now()).date()

    # get dates of all events
    event_dates = event_df[EVENT_DATE_COL].dt.date

    # get all events that have been hold
    event_select_df = event_df.loc[event_dates <= current_date]

    # get all event names
    return event_select_df[EVENT_NAME_COL].tolist()


def get_plot_features(session, plot_name):
    """
    get all plot features
    :param session: ff1 session object
    :param plot_name: name of plot
    :return: dictionary with input of each feature
    """
    # get list of features
    feature_lst = PLOT_FEATURES[plot_name]

    # iterate through all features
    output = dict()
    for feature in feature_lst:
        # select list of drivers
        if feature == DRIVER_NAMES:
            # get list of drivers
            driver_lst = session.results[ABBREVIATION_COL].to_list()

            # get selected drivers
            output[feature] = st.multiselect(feature, driver_lst, driver_lst[:3], help='Select Drivers')

        # select one driver
        elif feature == DRIVER_NAME:
            # get list of drivers
            driver_lst = session.results[ABBREVIATION_COL].to_list()

            output[feature] = st.selectbox(feature, options=driver_lst, index=0, help='Select Driver')

        # select drivers to compare
        elif feature == REF_DRIVER_NAMES:
            # get list of drivers
            driver_lst = session.results[ABBREVIATION_COL].to_list()

            # remove selected driver
            driver_lst.remove(output.get(DRIVER_NAME, ''))

            output[feature] = st.multiselect(feature, driver_lst, driver_lst[:2], help='Select Drivers')

        # flag if feature shall be shown
        elif feature in [SHOW_PERSONAL_BEST, SHOW_PIT_STOP, SHOW_TRACK_STATUS]:
            output[feature] = st.checkbox(feature, value=False)

        else:
            raise Exception('unknown plot name')

    return output
