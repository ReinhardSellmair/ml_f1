# helper functions to process results

import re

from defs import STATUS_COL


def get_classified(results_df):
    """
    get results of all drivers who finished the race
    :param results_df: results dataframe with column STATUS_COL
    :return: filtered results dataframe
    """
    def is_classified(s):
        if s == 'Finished' or re.findall(r'\+[1-5] Lap(?:s|)$', s):
            return True
        return False
    return results_df.loc[results_df[STATUS_COL].map(is_classified)]
