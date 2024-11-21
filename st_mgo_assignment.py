# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:08:58 2024

@author: carlo
"""

import streamlit as st
import datetime as dt
import requests
import json
import pandas as pd

@st.cache_data
def request_assignment(selected_date : str):
    base_url = 'https://mechanigo-auto-assignment-737521181245.asia-southeast1.run.app/assignment/?date='
    r = requests.get(f'{base_url}{selected_date}')
    
    return r

if 'selected_date' not in st.session_state:
    st.session_state['selected_date'] = dt.datetime.today().date().strftime('%Y-%m-%d')


selected_date = st.date_input(value = st.session_state['selected_date'],
                              format = "YYYY-MM-DD")

if st.button('Confirm'):
    if selected_date:
        
        r = request_assignment(selected_date)
        
        if r.status_code == 200:
            content = json.loads(r.content)['data']
            df = pd.DataFrame.from_dict(json.loads(r.content)['data'])
            
            st.dataframe(df)
            
            st.download_button(
                label = 'Download assignments.',
                data = df.to_csv(index = False),
                file_name = f"assignments_{selected_date}.csv",
                mime = "text/csv"
                )
            
        else:
            st.error(r.content)
        