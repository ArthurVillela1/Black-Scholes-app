import streamlit as st
import numpy as np
from scipy.stats import norm

def d1(S, K, r, T, sigma):
    return (np.log(S/K)+T*(r+np.pow(sigma, 2)/2))/(sigma*np.sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma) - vol*np.sqrt(T)

def call_value(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    return S*norm.cdf(d1_val)-K*np.exp(-r*T)*norm.cdf(d2_val)

def put_value(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    return K*np.exp(-r*T)*norm.cdf(-d2_val)-S*norm.cdf(-d1_val)

def delta(option_type, S, K, r, T, sigma):
    if option_type == "call":
        return norm.cdf(d1(S, K, r, T, sigma))
    elif option_type == "put":
        return norm.cdf(d1(S, K, r, T, sigma))-1

def gamma(S, K, r, T, sigma):
    return norm.cdf(d1(S, K, r, T, sigma))/S*sigma*sqrt(T)

def vega(S, K, r, T, sigma):
    return S*sqrt(T)*norm.cdf(d1(S, K, r, T, sigma))

def theta(option_type, S, K, r, T, sigma):
    if option_type == "call":
        return -S*norm.cdf(d1(S, K, r, T, sigma))*sigma/2*sqrt(T)-r*K*exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))
    elif option_type == "put":
        return -S*norm.cdf(d1(S, K, r, T, sigma))*sigma/2*sqrt(T)+r*K*exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))

def rho(option_type, S, K, r, T, sigma):
    if option_type == "call":
        return K*T*exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))
    elif option_type == "put":
        return -K*T*exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))

st.set_page_config(layout="wide")
st.title("Black-Scholes Pricing Model")
col1, col2 = st.columns(2)