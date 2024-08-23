import streamlit as st
import numpy as np
from scipy.stats import norm
import seaborn as sn
import matplotlib.pyplot as plt

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

with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Arthur Villela`</a>', unsafe_allow_html=True)

    current_price = st.number_input("Current Asset Price (S)", value=100.0)
    strike = st.number_input("Strike (K)", value=100.0)
    time_to_maturity = st.number_input("Years to Maturity (T)", value=1.0)
    volatility = st.number_input("Volatility (σ)", value=0.2)
    interest_rate = st.number_input("Risk-Free Interest Rate (r)", value=0.05)

    st.markdown("---")
    calculate_btn = st.button('Heatmap Parameters')
    spot_min = st.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
    spot_max = st.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
    vol_min = st.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
    vol_max = st.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)

    spot_range = np.linspace(spot_min, spot_max, 10)
    vol_range = np.linspace(vol_min, vol_max, 10)

def print_value(S, K, r, T, sigma):
    with col1:
        st.subheader("Call Value")
        st.title(f":green-background[{round(call_value(S, K, r, T, sigma), 2)}]")

    with col2:
        st.subheader("Put Value")
        st.title(f":red-background[{round(put_value(S, K, r, T, sigma), 2)}]")

cap = st.sidebar.number_input("Current Asset Price", value=80.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
sp = st.sidebar.number_input("Strike Price", value=100.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
ty = st.sidebar.number_input("Time to Maturity", value=1.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.4f")
vol = st.sidebar.number_input("Volatility", value=0.20, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
rfir = st.sidebar.number_input("Risk-Free Interest rate", value=0.05, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

print_value(cap, sp, rfir, ty, vol)