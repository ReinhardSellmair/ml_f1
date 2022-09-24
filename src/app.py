# streamlit app

import streamlit as st
import fastf1 as ff1

from defs import PAGE_TITLE
from streamlit_helper import get_seasons, get_race_event_names, get_plot_features
from laps_plots import LAP_PLOT_NAMES, create_figure

# set cache
ff1.Cache.enable_cache('res/cache')

# page settings
st.set_page_config(page_title=PAGE_TITLE, layout='wide')

# sidebar
with st.sidebar:

    # get list of available seasons
    seasons = get_seasons()

    # select season
    season = st.selectbox('Season', seasons, index=len(seasons) - 1, help='Select season')

    # get all races of selected season
    events = get_race_event_names(season)

    # select event
    event = st.selectbox('Race', events, index=len(events) - 1, help='Select race')

    # load session
    session = ff1.get_session(year=season, gp=event, identifier='R')
    try:
        session.load(telemetry=False, laps=True, weather=False)
    except ff1.api.SessionNotAvailableError:
        st.error('Data not available. Please choose other race.')

    # select plot
    plot_name = st.selectbox('Diagram', LAP_PLOT_NAMES, help='Select Diagram')

    # get features of plot
    feature_dict = get_plot_features(session, plot_name)

st.title(f'{season} {event}')
st.header(plot_name)

# create figure
fig = create_figure(session, plot_name, feature_dict)

# show figure
st.plotly_chart(fig, use_container_width=True)
