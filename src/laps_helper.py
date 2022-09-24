# helper functions for lap data

import pandas as pd

from defs import LAP_NUMBER_COL, TRACK_STATUS_COL, TRACK_STATUS_DF, PIT_IN_TIME_COL, PIT_OUT_TIME_COL, DRIVER_COL, \
    TIME_COL, STINT_COL, COMPOUND_COL,TEAM_COLOR_COL, UNKNOWN_COMPOUND_NAME, ABBREVIATION_COL, TEAM_COL
from utils import get_highest_hierarchy_ts, convert_timedelta_to_s


def get_highest_ts_per_lap(laps_df):
    """
    get highest hierarchy track status per lap
    :param laps_df: dataframe with lap information with columns LAP_NUMBER_COL, TRACK_STATUS_COL
    :return: dataframe with lap number and highest track status code
    """
    # concat all unique track status codes of the same lap
    lap_ts_df = laps_df.groupby(LAP_NUMBER_COL)[TRACK_STATUS_COL].apply(lambda s: ''.join(s.unique())).reset_index()
    # get highest hierarchy track status of each lap
    status_df = (pd.DataFrame([get_highest_hierarchy_ts(list(ts)) for ts in
                               lap_ts_df[TRACK_STATUS_COL]]).reset_index(drop=True))
    # attach lap number to status
    status_cols = list(status_df.columns)
    status_df[LAP_NUMBER_COL] = lap_ts_df[LAP_NUMBER_COL]
    return status_df[[LAP_NUMBER_COL] + status_cols]


def get_status_phases(laps_df):
    """
    get first and last lap of every track status phase
    :param laps_df: dataframe with lap number and track status code
    :return: dataframe with first and last lap of every track status phase
    """
    # get highest track status of each lap
    status_df = get_highest_ts_per_lap(laps_df)

    # create id of each phase
    status_df['phase_id'] = (status_df['code'].astype(int).diff() != 0).cumsum()
    # get first and last lap of each phase
    phase_df = status_df.groupby('phase_id').agg({LAP_NUMBER_COL: [min, max], 'code': 'first'})
    phase_df.columns = (phase_df
                        .columns.map('_'.join)
                        .to_series()
                        .map({'LapNumber_min': 'LapStart', 'LapNumber_max': 'LapEnd', 'code_first': 'code'}))
    # join track status and lap numbers
    phase_df = phase_df.set_index('code').join(TRACK_STATUS_DF.set_index('code')).sort_values('LapStart')
    return phase_df


def get_pit_laps(laps_df):
    """
    combine lap information of in and out lap of pit stops
    :param laps_df: dataframe with lap information with columns PIT_IN_TIME_COL, PIT_OUT_TIME_COL, LAP_NUMBER_COL,
                    DRIVER_COL, COMPOUND_COL, LAP_TIME_COL, TYRE_LIFE_COL
    :return: dataframe with pit in and pit out lap of each with stop with columns: DRIVER_COL, LAP_NUMBER_COL,
                    'PitTime', 'Compound_before', 'Compound_after', 'LapTime_before', 'LapTime_after',
                    'TyreLife_before', 'TyreLife_after'
    """
    # get pit in and pit out laps
    pit_in_df = laps_df.loc[~laps_df[PIT_IN_TIME_COL].isna()]
    pit_out_df = laps_df.loc[~laps_df[PIT_OUT_TIME_COL].isna()]

    # merge pit in and pit out laps
    pit_in_df = pit_in_df.assign(next_lap=pit_in_df[LAP_NUMBER_COL] + 1)
    pit_df = (pit_in_df.merge(pit_out_df, left_on=[DRIVER_COL, 'next_lap'], right_on=[DRIVER_COL, LAP_NUMBER_COL],
                              suffixes=('_before', '_after')))
    # calculate time in pit
    pit_df = pit_df.assign(PitTime_s=convert_timedelta_to_s(pit_df['PitOutTime_after'] - pit_df['PitInTime_before']))
    # select and rename columns
    pit_df.rename(columns={LAP_NUMBER_COL + '_before': LAP_NUMBER_COL, TEAM_COL + '_before': TEAM_COL}, inplace=True)
    pit_df.drop(columns=[LAP_NUMBER_COL + '_after', TEAM_COL + '_after'], inplace=True)

    return pit_df


def add_position(laps_df):
    """
    add driver position to each lap
    :param laps_df: dataframe with lap information with columns: TIME_COL, LAP_NUMBER_COL
    :return: lap information dataframe with additional column 'Position'
    """
    pos_df = (laps_df
              .sort_values(TIME_COL)
              .groupby(LAP_NUMBER_COL)
              .apply(lambda df: df.assign(Position=range(1, len(df)+1))))
    return pos_df


def calculate_gap_to_leader(laps_df):
    """
    calculate time gap to race leader in seconds after each lap
    :param laps_df: dataframe with lap information with columns TIME_COL, LAP_NUMBER_COL
    :return: lap information dataframe with attached columns: 'Position', 'GapToLeader_s'
    """
    # get driver positions
    laps_df = add_position(laps_df)

    # get leader of each lap
    leader_df = laps_df.loc[laps_df['Position'] == 1]

    # join lap start time of leader
    laps_df = (laps_df
               .set_index(LAP_NUMBER_COL)
               .join(leader_df.set_index(LAP_NUMBER_COL)[[TIME_COL]].rename(columns={TIME_COL: 'LeaderTime'}))
               .reset_index())
    # calculate gap to leader in s
    laps_df = laps_df.assign(GapToLeader_s=convert_timedelta_to_s(laps_df[TIME_COL] - laps_df['LeaderTime']))

    return laps_df


def get_stints(laps_df):
    """
    get first lap, last lap and compound of each drivers' stint (laps between pit stops)
    :param: laps_df: dataframe with lap information with columns: DRIVER_COL, STINT_COL, LAP_NUMBER_COL, COMPOUND_COL
    :return dataframe with first and last lap, and compound of each driver and stint
    """
    # get first and last lap of each stint
    stint_df = (laps_df
                .groupby([DRIVER_COL, TEAM_COL, STINT_COL])
                .agg({LAP_NUMBER_COL: ['min', 'max'], COMPOUND_COL: lambda x: x.iloc[0]}))
    col_name_map = {'LapNumber_min': 'FirstLap', 'LapNumber_max': 'LastLap', 'Compound_<lambda>': COMPOUND_COL}
    stint_df.columns = (stint_df.columns
                        .map('_'.join)
                        .to_series()
                        .map(col_name_map))

    # set first lap 1 to 0
    stint_df.loc[stint_df['FirstLap'] == 1, 'FirstLap'] = 0
    stint_df = stint_df.reset_index().sort_values('FirstLap')

    return stint_df


def join_team_color(df, session):
    """
    join team color to dataframe
    :param df: dataframe with required column DRIVER_COL
    :param session: fastf1 session object
    :return: dataframe with joined team color as column TEAM_COLOR_COL
    """

    # get driver number and team color
    team_color_df = session.results[[ABBREVIATION_COL, TEAM_COLOR_COL]].rename(columns={ABBREVIATION_COL: DRIVER_COL})
    # format color
    team_color_df[TEAM_COLOR_COL] = team_color_df[TEAM_COLOR_COL].map(lambda s: '#' + s)

    # add team color to dataframe
    df = df.set_index(DRIVER_COL).join(team_color_df.set_index(DRIVER_COL)).reset_index()

    return df


def fill_unknown_compound(laps_df):
    """
    replace unknown compound by compound with most laps of its stint
    :param laps_df: dataframe with laps information with columns COMPOUND_COL, DRIVER_COL, STINT_COL
    :return: dataframe with filled compound column
    """
    # get number of laps per stint and compound
    compound_stint_df = (laps_df
                         .loc[laps_df[COMPOUND_COL] != UNKNOWN_COMPOUND_NAME]
                         .groupby([DRIVER_COL, STINT_COL, COMPOUND_COL])
                         .size()
                         .reset_index()
                         .rename(columns={0: 'Laps'}))

    # get compound with highest number of laps per driver and stint
    compound_max_laps_df = (compound_stint_df
                            .loc[compound_stint_df.groupby([DRIVER_COL, STINT_COL])['Laps'].idxmax()])

    # join compound to laps
    laps_fill_df = (laps_df
                    .set_index([DRIVER_COL, STINT_COL])
                    .drop(COMPOUND_COL, axis=1)
                    .join(compound_max_laps_df
                          .set_index([DRIVER_COL, STINT_COL])[[COMPOUND_COL]])
                    .reset_index())
    # replace compound with compound_join if compound_join is not nan
    laps_fill_df.loc[laps_fill_df[COMPOUND_COL].isna(), COMPOUND_COL] = UNKNOWN_COMPOUND_NAME

    return laps_fill_df
