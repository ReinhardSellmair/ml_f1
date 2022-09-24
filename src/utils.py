# helper functions
from defs import TRACK_STATUS_DF

import pandas as pd
import numpy as np


def delta_to_timestamp(time_delta):
    """
    convert timedelta to timestamp
    :param time_delta: numpy time delta
    :return: pandas timestamp
    """
    return time_delta + pd.Timestamp("1970/01/01")


def lap_time_to_str(lap_time):
    """
    convert lap time to string
    :param lap_time: timestamp object
    :return: string with lap time
    """
    if pd.isnull(lap_time):
        return ""

    # check if lap time is type Timedelta
    if isinstance(lap_time, pd.Timedelta):
        # convert to timestamp
        lap_time = delta_to_timestamp(lap_time)

    return lap_time.strftime('%M:%S.%f')[:-3]


def convert_timedelta_to_s(timedelta):
    """
    get time delta in seconds
    :param timedelta: timedelta
    :return: float with time delta in seconds
    """
    return timedelta / np.timedelta64(1, 's')


def get_highest_hierarchy_ts(track_status_lst):
    """
    get highest hierarchy track status from list to track status codes
    :param track_status_lst: list of track status codes
    :return: highest hierarchy track status
    """
    # get all track status
    status_select_df = TRACK_STATUS_DF.loc[TRACK_STATUS_DF['code'].isin(track_status_lst)]
    # select highest hierarchy
    return status_select_df.sort_values('hierarchy').head(1).squeeze()


class KeyCounter:

    def __init__(self):
        self.memory = {}

    def get_key_count(self, primary_key, secondary_key):
        """
        get order when key pair has been entered the first time
        :param primary_key: any value
        :param secondary_key: any value
        :return: count when primary and secondary key pair has been provided the first time
        """
        if primary_key not in self.memory:
            count = 0
            self.memory[primary_key] = {secondary_key: count}
            return count
        elif secondary_key in self.memory[primary_key]:
            return self.memory[primary_key][secondary_key]
        else:
            count = len(self.memory[primary_key])
            self.memory[primary_key][secondary_key] = count
            return count


def get_current_year():
    """
    get current year
    :return: integer of current year
    """
    return pd.Timestamp.now().year
