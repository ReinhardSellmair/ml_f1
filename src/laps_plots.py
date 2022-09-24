# plots based on laps

import plotly.graph_objects as go
import pandas as pd
import numpy as np

from laps_helper import get_status_phases, join_team_color, fill_unknown_compound, get_pit_laps, add_position, \
    calculate_gap_to_leader, get_stints
from defs import DRIVER_COL, LAP_NUMBER_COL, TEAM_COLOR_COL, LAP_TIME_COL, PERSONAL_BEST_COL, TYRE_DF, \
    GRID_POSITION_COL, ABBREVIATION_COL, COMPOUND_COL, STINT_COL, TEAM_COL, STRAIGHT_SPEED_COL, PIT_IN_TIME_COL, \
    PIT_OUT_TIME_COL, POSITION_COL
from utils import delta_to_timestamp, KeyCounter, convert_timedelta_to_s, lap_time_to_str
from results_helper import get_classified

# height of figure
DEFAULT_HEIGHT = 500

# plot names
LAP_TIME_PLOT_NAME = 'Lap Times'
LAP_TIME_DIFF_PLOT_NAME = 'Lap Time Difference'
POSITION_PLOT_NAME = 'Driver Position'
LEADER_GAP_PLOT_NAME = 'Gap to Leader'
STINT_PLOT_NAME = 'Stints'
PIT_TIME_PLOT_NAME = 'Pit Times'
FASTEST_LAP_PLOT_NAME = 'Fastest Lap'
TOP_SPEED_PLOT_NAME = 'Top Speed'
LAP_PLOT_NAMES = [LAP_TIME_PLOT_NAME, LAP_TIME_DIFF_PLOT_NAME, POSITION_PLOT_NAME, LEADER_GAP_PLOT_NAME,
                  STINT_PLOT_NAME, PIT_TIME_PLOT_NAME, FASTEST_LAP_PLOT_NAME, TOP_SPEED_PLOT_NAME]

# feature names
DRIVER_NAMES = 'Drivers'
DRIVER_NAME = 'Driver'
REF_DRIVER_NAMES = 'Reference Drivers'
SHOW_PERSONAL_BEST = 'Show Personal Best'
SHOW_PIT_STOP = 'Show Pit Stop'
SHOW_TRACK_STATUS = 'Show Track Status'

# features of each plot
PLOT_FEATURES = {LAP_TIME_PLOT_NAME: [DRIVER_NAMES, SHOW_PERSONAL_BEST, SHOW_PIT_STOP, SHOW_TRACK_STATUS],
                 LAP_TIME_DIFF_PLOT_NAME: [DRIVER_NAME, REF_DRIVER_NAMES, SHOW_PERSONAL_BEST, SHOW_PIT_STOP,
                                           SHOW_TRACK_STATUS],
                 POSITION_PLOT_NAME: [SHOW_PIT_STOP, SHOW_TRACK_STATUS],
                 LEADER_GAP_PLOT_NAME: [SHOW_PIT_STOP, SHOW_TRACK_STATUS],
                 STINT_PLOT_NAME: [SHOW_TRACK_STATUS],
                 PIT_TIME_PLOT_NAME: [],
                 FASTEST_LAP_PLOT_NAME: [],
                 TOP_SPEED_PLOT_NAME: []}

# placeholder to insert column value to hover text
VAL_PLACEHOLDER = '<col_val>'


def create_figure(session, plot_name, feature_dict):
    """
    create plotly figure of selected plot
    :param session: fastf1 session object
    :param plot_name: name of plot
    :param feature_dict: dictionary with feature values
    :return: plotly figure
    """
    if plot_name == LAP_TIME_PLOT_NAME:
        fig = plot_lap_times(session, driver_lst=feature_dict[DRIVER_NAMES],
                             show_personal_best=feature_dict[SHOW_PERSONAL_BEST],
                             show_pit_stop=feature_dict[SHOW_PIT_STOP],
                             show_track_status=feature_dict[SHOW_TRACK_STATUS])
    elif plot_name == LAP_TIME_DIFF_PLOT_NAME:
        fig = plot_lap_time_diff(session, ref_driver=feature_dict[DRIVER_NAME],
                                 driver_lst=feature_dict[REF_DRIVER_NAMES],
                                 show_personal_best=feature_dict[SHOW_PERSONAL_BEST],
                                 show_pit_stop=feature_dict[SHOW_PIT_STOP],
                                 show_track_status=feature_dict[SHOW_TRACK_STATUS])
    elif plot_name == POSITION_PLOT_NAME:
        fig = plot_driver_position(session, show_pit_stop=feature_dict[SHOW_PIT_STOP],
                                   show_track_status=feature_dict[SHOW_TRACK_STATUS])
    elif plot_name == LEADER_GAP_PLOT_NAME:
        fig = plot_gap_to_leader(session, show_pit_stop=feature_dict[SHOW_PIT_STOP],
                                 show_track_status=feature_dict[SHOW_TRACK_STATUS])
    elif plot_name == STINT_PLOT_NAME:
        fig = plot_stints(session, show_track_status=feature_dict[SHOW_TRACK_STATUS])
    elif plot_name == PIT_TIME_PLOT_NAME:
        fig = plot_pit_times(session)
    elif plot_name == FASTEST_LAP_PLOT_NAME:
        fig = plot_fastest_lap(session)
    elif plot_name == TOP_SPEED_PLOT_NAME:
        fig = plot_top_speed(session)
    else:
        raise ValueError('Plot name not found: {}'.format(plot_name))

    # adjust height
    fig.update_layout(height=DEFAULT_HEIGHT)

    return fig


def plot_track_status(session, max_hierarchy=5):
    """
    Create lap plot where track status phases like e.g. safety car are highlighted
    :param session: fastf1 session object of selected race
    :param max_hierarchy: integer of lowest track status to be shown
    :return: plotly figure object
    """
    fig = go.Figure()

    # get laps
    laps_df = session.laps

    # get track phases
    track_phase_df = get_status_phases(laps_df)
    track_phase_select_df = track_phase_df.loc[track_phase_df['hierarchy'] <= max_hierarchy]

    # add rectangles for each phase to figure
    for _, row in track_phase_select_df.iterrows():
        fig.add_vrect(row['LapStart'], row['LapEnd'] + 1, fillcolor=row['color'], line_width=0,
                      annotation_text=row['name'], opacity=0.5, annotation_textangle=270,
                      annotation_position="top left")

    return fig


def get_customdata_hovertemplate(plot_df, driver_col=DRIVER_COL, team_col=TEAM_COL, lap_number_col=LAP_NUMBER_COL,
                                 lap_time_col=LAP_TIME_COL, compound_col=COMPOUND_COL, extra_cols=dict(), title=None):
    """
    Create custom data and hover template for plotly figures
    :param plot_df: dataframe with values to be shown in hover text
    :param driver_col: name of driver column (if None, no driver name is shown)
    :param team_col: name of team column (if None, no team name is shown)
    :param lap_number_col: name of lap number column (if None, no lap number is shown)
    :param lap_time_col: name of lap time column (if None, no lap time is shown)
    :param compound_col: name of compound column (if None, no compound is shown)
    :param extra_cols: dictionary with additional columns to be shown in hover text (key: column name, value: hover
                       text) (column value is inserted with placeholder VAL_PLACEHOLDER)
    :param title: title of hover text (if None, no title is shown)
    :return: array with custom data and hover template string
    """
   # text of standard columns
    standard_text = [f'Driver: {VAL_PLACEHOLDER}', f'Team: {VAL_PLACEHOLDER}', f'Lap: {VAL_PLACEHOLDER}',
                     f'Lap Time: {VAL_PLACEHOLDER}', f'Compound: {VAL_PLACEHOLDER}']

    # create dictionary to map column names to hover text
    col_to_text = {}
    for col, text in zip([driver_col, team_col, lap_number_col, lap_time_col, compound_col], standard_text):
        if col:
            col_to_text[col] = text

    # add extra columns to dictionary
    col_to_text.update(extra_cols)

    # get custom data
    custom_data = plot_df[list(col_to_text.keys())].values

    # check if title is given
    if title:
        hover_template = "<b>" + title + "</b><br><br>"
    else:
        hover_template = ""

    # add hover text for each column
    for i, text in enumerate(col_to_text.values()):
        hover_template += text.replace(VAL_PLACEHOLDER, f'%{{customdata[{i}]}}') + "<br>"

    # add extra flag
    hover_template += "<extra></extra>"

    return custom_data, hover_template


def plot_personal_best(fig, plot_df, y_col):
    """
    mark driver's best lap
    :param fig: plotly figure
    :param plot_df: dataframe with driver laps
    :param y_col: column to be shown on y-axis
    :return: plotly figure with driver's personal best lap
    """
    best_lap_df = plot_df.loc[plot_df[PERSONAL_BEST_COL]].iloc[0]

    # add column with lap time as string
    best_lap_df['lap_time_str'] = best_lap_df[LAP_TIME_COL].strftime('%M:%S.%f')[:-3]

    # get custom data and hover template
    custom_data, hover_template = get_customdata_hovertemplate(best_lap_df, lap_time_col='lap_time_str',
                                                               title='Personal Best')

    fig.add_trace(
        go.Scatter(x=[best_lap_df[LAP_NUMBER_COL]], y=[best_lap_df[y_col]], showlegend=False,
                   marker={'color': best_lap_df[TEAM_COLOR_COL], 'size': 10}, name='Personal Best',
                   hovertemplate=hover_template, customdata=[custom_data]))

    return fig


def plot_pit_stop(fig, plot_df, y_col):
    """
    plot driver pit stops
    :param fig: plotly figure object
    :param plot_df: laps dataframe of one driver
    :param y_col: name of column to be plotted on y axis
    :return: plotly figure object with added traces
    """
    # get color
    color = plot_df[TEAM_COLOR_COL].values[0]

    # get laps when driver pitted
    pit_df = get_pit_laps(plot_df)
    # join color to compound after stop
    pit_df = pit_df.merge(TYRE_DF, how='left', left_on='Compound_after', right_on='Compound')
    for _, stop in pit_df.iterrows():
        # get custom data and hover template
        custom_data, hover_template = \
            get_customdata_hovertemplate(stop, lap_time_col=None, compound_col='Compound_after',
                                         extra_cols={'PitTime_s': f'Pit Time: {VAL_PLACEHOLDER}s'},
                                         title='Pit Stop')

        marker_dict = {'color': stop['color'], 'size': 10, 'symbol': 'square',
                       'line': {'color': color, 'width': 2}}
        fig.add_trace(
            go.Scatter(x=[stop[LAP_NUMBER_COL]], y=[stop[y_col]],
                       showlegend=False, mode='markers', marker=marker_dict, hovertemplate=hover_template,
                       customdata=[custom_data], hoverlabel={'bgcolor': color}, name=''))

    return fig


def plot_lap(session, laps_df, driver_lst, y_col, show_track_status, show_personal_best, show_pit_stop,
             hover_add_dict=dict()):
    """
    plot laps
    :param session: fastf1 session object
    :param laps_df: lap time dataframe with column y_col
    :param driver_lst: list of drivers to be plotted
    :param y_col: dataframe column to be plotted on y-axis
    :param show_track_status: flag if track status shall be shown
    :param show_personal_best: flag if drivers' personal best lap shall be highlighted
    :param show_pit_stop: flag if drivers' pit stops shall be highlighted
    :param hover_add_dict: dictionary with additional columns to be shown in hover text (key: column name, value: hover
                           text)
    :return: plotly figure object
    """
    # fill missing compound information
    laps_df = fill_unknown_compound(laps_df)

    # check if track status shall be plotted
    if show_track_status:
        fig = plot_track_status(session)
    else:
        fig = go.Figure()

    key_counter = KeyCounter()
    for driver in driver_lst:
        plot_df = laps_df.loc[laps_df[DRIVER_COL] == driver].sort_values(LAP_NUMBER_COL)
        # get color
        color = plot_df[TEAM_COLOR_COL].values[0]

        # check if color has been used already
        if key_counter.get_key_count(color, driver) == 0:
            line_dict = {'color': color}
        else:
            # plot dashed line
            line_dict = {'color': color, 'dash': 'dash'}

        # add column with lap time as string
        plot_df['lap_time_str'] = plot_df[LAP_TIME_COL].map(lap_time_to_str)

        # get custom data and hover template
        custom_data, hover_template = get_customdata_hovertemplate(plot_df, lap_time_col='lap_time_str',
                                                                   extra_cols=hover_add_dict)

        fig.add_trace(
            go.Scatter(x=plot_df[LAP_NUMBER_COL], y=plot_df[y_col], mode='lines', name=driver, line=line_dict,
                       customdata=custom_data, hovertemplate=hover_template, hoverlabel={'bgcolor': color}))

        if show_personal_best:
            fig = plot_personal_best(fig, plot_df, y_col)

        if show_pit_stop:
            fig = plot_pit_stop(fig, plot_df, y_col + '_before')

    fig.update_xaxes(title='Lap', range=[0.5, laps_df[LAP_NUMBER_COL].max() + 0.5])

    return fig


def plot_lap_times(session, driver_lst, show_personal_best=False, show_pit_stop=False, show_track_status=False):
    """
    plot lap times of selected drivers
    :param session: fastf1 session object of selected race
    :param driver_lst: list of driver abbreviations to be shown
    :param show_personal_best: flag if drivers' personal best lap shall be highlighted
    :param show_pit_stop: flag if drivers' pit stops shall be highlighted
    :param show_track_status: flag if track status shall be shown
    :return: plotly fig object
    """
    # select lap times of drivers
    laps_df = session.laps.loc[session.laps[DRIVER_COL].isin(driver_lst)]

    # join team color
    laps_df = join_team_color(laps_df, session)

    # convert lap times to timestamps
    laps_df[LAP_TIME_COL] = delta_to_timestamp(laps_df[LAP_TIME_COL])

    fig = plot_lap(session, laps_df, driver_lst, LAP_TIME_COL, show_track_status, show_personal_best, show_pit_stop)

    fig.update_layout(yaxis_tickformat='%M:%S.%f')
    fig.update_yaxes(title='Lap Time')

    return fig


def plot_lap_time_diff(session, ref_driver, driver_lst, show_track_status=True, show_personal_best=True,
                       show_pit_stop=True):
    """
    plot lap time difference to reference driver
    :param session: fastf1 session object of selected race
    :param ref_driver: abbreviation of reference driver (lap time difference of this driver will be shown as 0)
    :param driver_lst: list driver abbreviations to compare against reference driver
    :param show_track_status: flag if track status shall be shown
    :param show_personal_best: flag if drivers' personal best lap shall be highlighted
    :param show_pit_stop: flag if drivers' pit stops shall be highlighted
    :return: plotly figure object with plot
    """
    # select lap times of reference driver
    ref_laps_df = session.laps.loc[session.laps[DRIVER_COL] == ref_driver]

    # get laps of selected drivers
    laps_df = session.laps.loc[session.laps[DRIVER_COL].isin(driver_lst + [ref_driver])]

    # calculate lap time difference to reference driver
    laps_diff_df = (laps_df
                    .set_index(LAP_NUMBER_COL)
                    .join(ref_laps_df.set_index(LAP_NUMBER_COL)[[LAP_TIME_COL]]
                          .rename(columns={LAP_TIME_COL: 'RefLapTime'}))
                    .reset_index())
    laps_diff_df = (laps_diff_df
                    .assign(LapTimeDiff_s=convert_timedelta_to_s(laps_diff_df[LAP_TIME_COL] -
                                                                 laps_diff_df['RefLapTime'])))

    # join team color
    laps_diff_df = join_team_color(laps_diff_df, session)
    # convert lap times to timestamps
    laps_diff_df[LAP_TIME_COL] = delta_to_timestamp(laps_diff_df[LAP_TIME_COL])

    # columns to be added to hover text
    hover_add_dict = {'LapTimeDiff_s': f'Difference: {VAL_PLACEHOLDER} s'}

    fig = plot_lap(session, laps_diff_df, [ref_driver] + driver_lst, 'LapTimeDiff_s', show_track_status,
                   show_personal_best, show_pit_stop, hover_add_dict=hover_add_dict)

    fig.update_layout(title=f'Lap Time Difference to {ref_driver}')
    fig.update_yaxes(title='Lap Time Difference in s')

    return fig


def plot_driver_position(session, show_track_status=True, show_pit_stop=True):
    """
    plot driver position
    :param session: fastf1 session object of selected race
    :param show_pit_stop: flag if track status shall be shown
    :param show_track_status: flag if drivers' pit stops shall be highlighted
    :return: plotly figure object with plot
    """
    # get results
    results_df = session.results
    # correct 0 grid position
    results_df.loc[results_df[GRID_POSITION_COL] == 0, GRID_POSITION_COL] = results_df[GRID_POSITION_COL].max() + 1

    # get laps
    laps_df = add_position(session.laps)
    # number of laps
    n_laps = laps_df[LAP_NUMBER_COL].max()

    # add grid and result positions
    grid_df = (results_df[[ABBREVIATION_COL, GRID_POSITION_COL]]
               .rename(columns={ABBREVIATION_COL: DRIVER_COL, GRID_POSITION_COL: POSITION_COL})
               .assign(LapNumber=0, Stint=1)
               .set_index(DRIVER_COL)
               .join(laps_df.loc[laps_df[LAP_NUMBER_COL] == 1, [DRIVER_COL, COMPOUND_COL]].set_index(DRIVER_COL))
               .reset_index())
    classified_df = get_classified(results_df)
    final_df = (classified_df[[ABBREVIATION_COL, POSITION_COL]]
                .rename(columns={ABBREVIATION_COL: DRIVER_COL})
                .assign(LapNumber=n_laps + 1)
                .set_index(DRIVER_COL)
                .join(laps_df.loc[laps_df.groupby(DRIVER_COL)[LAP_NUMBER_COL].idxmax()]
                      .set_index(DRIVER_COL)[[COMPOUND_COL, STINT_COL]])
                .reset_index())

    col_select = [DRIVER_COL, LAP_NUMBER_COL, POSITION_COL, COMPOUND_COL, STINT_COL, PIT_IN_TIME_COL, PIT_OUT_TIME_COL,
                  TEAM_COL, LAP_TIME_COL]
    position_df = (pd.concat([grid_df, laps_df[col_select], final_df]).reset_index())

    # join team color
    position_df = join_team_color(position_df, session)

    # add row to hover text
    hover_add_dict = {POSITION_COL: f'Position: {VAL_PLACEHOLDER}'}

    fig = plot_lap(session, position_df, results_df[ABBREVIATION_COL], POSITION_COL, show_track_status,
                   False, show_pit_stop, hover_add_dict=hover_add_dict)

    # define xticks
    x_tickvals = np.unique(np.append(np.arange(0, n_laps, 5), np.array([n_laps + 1])))
    x_ticktext = x_tickvals.astype(str)
    x_ticktext[0] = 'Grid'
    x_ticktext[-1] = 'Final Result'

    fig.update_yaxes(title=POSITION_COL, range=[laps_df[POSITION_COL].max() + 0.5, 0.5])
    fig.update_xaxes(title='Lap', range=[-0.5, n_laps + 1.5])
    fig.update_layout(yaxis={'tickvals': [1, 5, 10, 15, 20]}, xaxis={'tickvals': x_tickvals, 'ticktext': x_ticktext})

    return fig


def plot_gap_to_leader(session, show_pit_stop=True, show_track_status=True):
    """
    plot time difference to leader
    :param session: fastf1 session object of selected race
    :param show_pit_stop: flag if track status shall be shown
    :param show_track_status: flag if drivers' pit stops shall be highlighted
    :return: plotly figure object with plot
    """

    # get laps
    laps_df = session.laps
    # calculate gap to leader
    laps_df = calculate_gap_to_leader(laps_df)

    # get list of drivers
    driver_lst = session.results[ABBREVIATION_COL]

    # join team color
    laps_df = join_team_color(laps_df, session)

    # add row to hover text
    hover_add_dict = {'GapToLeader_s': f'Gap to Leader: {VAL_PLACEHOLDER} s'}

    fig = plot_lap(session, laps_df, driver_lst, 'GapToLeader_s', show_track_status,
                   False, show_pit_stop, hover_add_dict=hover_add_dict)

    fig.update_layout(title='Gap to Leader')
    fig.update_yaxes(title='Gap in s', range=[laps_df['GapToLeader_s'].max() + 1, -5])

    return fig


def plot_stints(session, show_track_status=False):
    """
    plot tyre choice of each driver
    :param session: fastf1 session object of selected race
    :param show_track_status: flag if drivers' pit stops shall be highlighted
    :return: plotly figure object with plot
    """
    # get laps
    laps_df = session.laps
    laps_df = fill_unknown_compound(laps_df)

    # get results
    results_df = session.results.copy()
    # format team color
    results_df[TEAM_COLOR_COL] = results_df[TEAM_COLOR_COL].map(lambda s: '#' + s)

    # get stints
    stints_df = get_stints(laps_df)
    # join driver position and tyre color
    stints_df = (stints_df
                 .set_index(DRIVER_COL)
                 .join(results_df
                       .rename(columns={ABBREVIATION_COL: DRIVER_COL})
                       .set_index(DRIVER_COL)[[POSITION_COL, TEAM_COLOR_COL]])
                 .reset_index()
                 .set_index(COMPOUND_COL)
                 .join(TYRE_DF.set_index(COMPOUND_COL))
                 .rename(columns={'color': 'TyreColor'})
                 .reset_index()
                 .sort_values([POSITION_COL, STINT_COL]))

    if show_track_status:
        fig = plot_track_status(session)
    else:
        fig = go.Figure()

    compound_set = set()
    for _, stint in stints_df.iterrows():
        if stint[COMPOUND_COL] in compound_set:
            show_legend = False
        else:
            compound_set.add(stint[COMPOUND_COL])
            show_legend = True

        # get cutomdata and hovertext
        extra_cols = {'FirstLap': f'First Lap: {VAL_PLACEHOLDER}', 'LastLap': f'Last Lap: {VAL_PLACEHOLDER}'}
        customdata, hover_template = get_customdata_hovertemplate(stint, lap_number_col=None, lap_time_col=None,
                                                             extra_cols=extra_cols)

        fig.add_trace(go.Bar(y=[stint[DRIVER_COL]], x=[stint['LastLap'] - stint['FirstLap'] + 1], orientation='h',
                             marker={'color': stint['TyreColor'], 'line_width': 3}, name=stint[COMPOUND_COL],
                             showlegend=show_legend, customdata=[customdata], hovertemplate=hover_template,
                             hoverlabel={'bgcolor': stint[TEAM_COLOR_COL]}))

    fig.update_layout(barmode='stack')
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title='Lap')

    return fig


def plot_vertical_bar(plot_df, x_col, hover_add_dict=dict(), reversed=False, compound_col=COMPOUND_COL,
                      lap_time_col=LAP_TIME_COL):
    """
    plot vertical bars where each bar represents a driver
    :param plot_df: dataframe to be plotted with columns: x_col, TEAM_COLOR_COL, DRIVER_COL, TEAM_COL, LAP_NUMBER_COL
    :param x_col: name of column to be plotted on x-axis
    :param hover_add_dict: dictionary with to map additional columns to labels to be shown in hover text
    :param reversed: flag if order of bars shall be reversed
    :param compound_col: name of column with tyre compound (if None, no compound will be shown)
    :param lap_time_col: name of column with lap time (if None, no lap time will be shown)
    :return: plotly figure object
    """

    # assign position
    plot_df = plot_df.sort_values(x_col).assign(Position=range(1, len(plot_df) + 1))

    fig = go.Figure()

    key_counter = KeyCounter()
    for _, row in plot_df.iterrows():
        # get pattern of bar
        if key_counter.get_key_count(row[TEAM_COLOR_COL], row[DRIVER_COL]) == 0:
            pattern = ''
        else:
            pattern = '/'

        # get cutomdata and hovertext
        customdata, hover_template = get_customdata_hovertemplate(row, extra_cols=hover_add_dict,
                                                                  compound_col=compound_col, lap_time_col=lap_time_col)

        fig.add_trace(go.Bar(y=[row[POSITION_COL]], x=[row[x_col]], orientation='h',
                             marker={'color': row[TEAM_COLOR_COL]}, name='', showlegend=False,
                             hoverlabel={'bgcolor': row[TEAM_COLOR_COL]}, customdata=[customdata],
                             hovertemplate=hover_template, marker_pattern_shape=pattern))

    if reversed:
        fig.update_yaxes(autorange="reversed")

    fig.update_layout(yaxis={'ticktext': plot_df[DRIVER_COL], 'tickvals': plot_df[POSITION_COL]})
    fig.update_xaxes(title='Pit in time in s')

    fig.update_layout(height=len(plot_df) * 15)

    return fig


def plot_pit_times(session):
    """
    plot pit stop times of each stop
    :param session: fastf1 session object
    :return: plotly figure object
    """

    # get laps and results from session
    laps_df = session.laps

    # get pit stops
    pit_df = get_pit_laps(laps_df)

    # join team color
    pit_df = join_team_color(pit_df, session)

    # extra rows to be shown in hover text
    hover_add_dict = {'PitTime_s': f'Time: {VAL_PLACEHOLDER} s'}

    fig = plot_vertical_bar(pit_df, 'PitTime_s', reversed=True, hover_add_dict=hover_add_dict, compound_col=None,
                            lap_time_col=None)

    fig.update_xaxes(title='Pit in time in s')

    return fig


def plot_fastest_lap(session):
    """
    plot fastest lap of each driver
    :param session: fastf1 session object
    :return: plotly figure object
    """
    # get laps and results
    laps_df = fill_unknown_compound(session.laps)

    # get fastest lap of each driver
    best_lap_df = laps_df.loc[laps_df[PERSONAL_BEST_COL]].sort_values(LAP_TIME_COL)
    # get fastest lap
    fastest_lap = best_lap_df.iloc[0]
    # calculate gap to fastest lap
    best_lap_df = (best_lap_df
                   .assign(LapTime_Diff_s=convert_timedelta_to_s(best_lap_df[LAP_TIME_COL] -
                                                                 fastest_lap[LAP_TIME_COL])))

    # convert lap times to strings
    best_lap_df[LAP_TIME_COL] = best_lap_df[LAP_TIME_COL].map(lap_time_to_str)

    # join team color
    best_lap_df = join_team_color(best_lap_df, session)

    hover_add_dict = {'LapTime_Diff_s': f'Time Gap: {VAL_PLACEHOLDER} s'}
    fig = plot_vertical_bar(best_lap_df, 'LapTime_Diff_s', hover_add_dict=hover_add_dict, reversed=True)

    fig.update_xaxes(title='Gap to Fastest Lap in s')

    return fig


def plot_top_speed(session):
    """
    plot vertical bars of each drivers' top speed
    :param session: fastf1 session object
    :return: plotly figure object
    """
    # get laps and results
    laps_df = session.laps

    # get highest speed of each driver
    high_speed_df = (laps_df
                     .loc[(laps_df
                           .loc[~laps_df[STRAIGHT_SPEED_COL].isna()]
                           .groupby(DRIVER_COL)[STRAIGHT_SPEED_COL]
                           .idxmax())]
                     .sort_values('SpeedST'))

    # add team colors
    high_speed_df = join_team_color(high_speed_df, session)


    hover_add_dict = {STRAIGHT_SPEED_COL: f'Top Speed: {VAL_PLACEHOLDER} km/h'}
    fig = plot_vertical_bar(high_speed_df, STRAIGHT_SPEED_COL, hover_add_dict=hover_add_dict, lap_time_col=None)

    fig.update_xaxes(title='Top Speed in km/h')

    return fig
