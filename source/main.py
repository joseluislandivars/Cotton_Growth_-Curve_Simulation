#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit main function
"""
from matplotlib import container
import pandas as pd
import streamlit as st

from utility_functions import five_params_func
from utility_functions import params_search
from utility_functions import predict_func
from utility_functions import derivative_func
from utility_functions import plot_func

def main():
    """Strea Frame"""
    st.set_page_config(layout="wide")
    
    # sidebar
    with st.sidebar:
        # title and formula
        st.header("Growth Curve")
        st.latex(r"y = d + \frac{a -d }{(1 + (\frac{x}{c})^b)^g}")

        # upload data
        uploaded_file = st.file_uploader("choose a file")

        # range of the day 
        range_values = st.slider(
                "select a range: ", 1, 140, (0, 140)
                )
       
    # load data
    if uploaded_file:
        if uploaded_file.name.split(".")[-1] == "xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        num_columns = len(df.columns) - 1
        selected_curve = st.slider("curve: ", 1, num_columns, 1)

        x_value_index = df["x"].notna()
        x_values = df["x"][x_value_index].values
        y_values = df.iloc[:, selected_curve][x_value_index].values

        # solve parameters
        params = params_search(five_params_func, x_values, y_values)
        
        fig, max_day, max_growth_rate, sec_max_day, sec_min_day = plot_func(params, x_values, y_values, range_values[0], range_values[1])

        # show results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("selected curve:", df.columns[selected_curve])
        with col2:
            st.metric("max growth rate day", int(max_day))
        with col3:
            st.metric("max growth rate", max_growth_rate)
        with col4:
            st.metric("second derivative max day", int(sec_max_day))
            st.metric("second derivative min day", int(sec_min_day))
        
        # plot 
        st.pyplot(fig)

        # save predicted values 


        # save params
        
        with st.sidebar:
            st.button("hello world")


    else:
        st.title("magic!")


if __name__ == "__main__":
    main()





