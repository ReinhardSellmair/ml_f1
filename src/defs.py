# definitions

import pandas as pd

# track status
TRACK_STATUS_DF = pd.DataFrame({'code': ['1', '2', '4', '5', '6', '7'],
                                'name': ['All Clear', 'Yellow', 'SC', 'Red', 'VSC Deployed', 'VSC Ending'],
                                'hierarchy': [6, 5, 2, 1, 3, 4],
                                'color': [None, '#ffff00', '#ff9300', '#ff0000', '#ffc900', '#ffc900']})

# tyre
UNKNOWN_COMPOUND_NAME = 'UNKNOWN'
TYRE_DF = pd.DataFrame({'Compound': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET', UNKNOWN_COMPOUND_NAME],
                        'color': ['#ed2029', '#fcd50d', '#ffffff', '#4db849', '#457ec2', '#b2b6ba']})

# laps columns
COMPOUND_COL = 'Compound'
DRIVER_NUM_COL = 'DriverNumber'
STINT_COL = 'Stint'
TRACK_STATUS_COL = 'TrackStatus'
LAP_NUMBER_COL = 'LapNumber'
PIT_IN_TIME_COL = 'PitInTime'
PIT_OUT_TIME_COL = 'PitOutTime'
DRIVER_COL = 'Driver'
LAP_TIME_COL = 'LapTime'
TYRE_LIFE_COL = 'TyreLife'
TIME_COL = 'Time'
PERSONAL_BEST_COL = 'IsPersonalBest'
TEAM_COL = 'Team'
STRAIGHT_SPEED_COL = 'SpeedST'

# results columns
STATUS_COL = 'Status'
TEAM_COLOR_COL = 'TeamColor'
GRID_POSITION_COL = 'GridPosition'
ABBREVIATION_COL = 'Abbreviation'
POSITION_COL = 'Position'

# event columns
EVENT_DATE_COL = 'EventDate'
EVENT_NAME_COL = 'EventName'

# streamlit
PAGE_TITLE = 'F1 Analytics'
FIRST_SEASON = 2018
