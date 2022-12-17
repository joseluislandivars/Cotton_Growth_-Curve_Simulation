#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import sympy 
from scipy.optimize import curve_fit

import streamlit as st


def five_params_func(x, a, b, c, d, g):
    """Five parameters logistic function"""
    return d + (a - d)/(1 + (x/c)**b)**g


def params_search(func, x_values, y_values, method="trf", maxfev=1000000):
    """Search for the parameters' values """
    try:
        params, _ = curve_fit(func, x_values, y_values, method=method, maxfev=maxfev)
    except RuntimeError:
        print("cannot be solved")
    else:
        return params

def predict_func(func, params, lower, upper):
    """Predict the values for selected range"""
    range_value = upper - lower + 1
    x_values = np.linspace(lower, upper, range_value)
    return func(x_values, *params)

def derivative_func(params, lower, upper, derivative=1):
    """Generate the first derivative values for selected range"""
    range_value = upper - lower + 1 
    x_values = np.linspace(lower, upper, range_value)

    # symbolic variable
    x = sympy.Symbol('x')
    a, b, c, d, g = params
    func = d + (a - d)/(1 + (x/c)**b)**g
    first_deriv = func.diff()

    if derivative == 2:
        second_deriv = first_deriv.diff()
        return np.array([second_deriv.subs({x: x_value}) for x_value in x_values])

    return np.array([first_deriv.subs({x: x_value}) for x_value in x_values])
    
def plot_func(params, x_values, y_values, lower, upper):
    """Plot curve"""
    # original points
    x_value_index = [int(np.where(x_values == x)[0]) for x in x_values if lower <= x <= upper]
    x_values = x_values[x_value_index]
    y_values = y_values[x_value_index]

    # predicted value curve
    range_value = upper - lower + 1
    x = np.linspace(lower, upper, range_value)
    predicted_values = predict_func(five_params_func, params, lower, upper)

    # first derivative 
    first_deriv_predicted_values = derivative_func(params, lower, upper)
    max_x_index = np.argmax(first_deriv_predicted_values)

    max_day = x[max_x_index]
    max_growth_rate = round(float(first_deriv_predicted_values[max_x_index]), 3)

    # second derivative
    second_derive_predicted_values = derivative_func(params, lower, upper, 2)
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
    # plot orginal data
    axes[0].scatter(x_values, y_values, color="red", label="original data")
    # plot growth curve 
    axes[0].plot(x, predicted_values, label="growth curve")
    axes[0].plot([x[max_x_index], x[max_x_index]], [0,  predicted_values[max_x_index]], linestyle="--", label="max growth rate day")
    axes[0].legend()

    # plot first derivative
    axes[1].plot(x, first_deriv_predicted_values, label="first deriv")

    # plot second derivative
    axes[2].plot(x, second_derive_predicted_values, label="first deriv")
    
    
    # ax.plot([max_index, max_index], [0, predicted_values[max_index]], linestyle="--", color="orange", label="growth curve")

    # plot the predicted curve

    # plt.show()
    return fig, max_day, max_growth_rate

def save_predicted_values():
    pass

def save_params():
    pass




if __name__ == "__main__":
    dataset_path = "../dataset/growth_curve_dataset.xlsx"
    dataset = pd.read_excel(dataset_path)
    
    x_value_index =  dataset['x'].notna()

    x_values = dataset['x'][x_value_index].values
    y_values = dataset.iloc[:, 1][x_value_index].values
    
    assert len(x_values) == len(y_values), "not equal length"

    # params, _ = curve_fit(five_params_func, x_values, y_values, method="trf", maxfev=100000)
    params = params_search(five_params_func,  x_values, y_values, "trf", 1000000)

    predicted_values = predict_func(five_params_func, params, 10, 30)

    test = derivative_func(params, 10, 30, 2)

    plot_func(params, x_values, y_values, 1, 140)




