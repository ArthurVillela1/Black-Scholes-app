import streamlit as st
import numpy as np
from scipy.stats import norm
import seaborn as sn
import matplotlib.pyplot as plt

def d1(S, K, r, T, sigma):
    return (np.log(S/K)+T*(r+np.pow(sigma, 2)/2))/(sigma*np.sqrt(T))

def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma) - sigma*np.sqrt(T)

def call_value(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    BS_call = S*norm.cdf(d1_val)-K*np.exp(-r*T)*norm.cdf(d2_val)
    return BS_call

def put_value(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    BS_put = K*np.exp(-r*T)*norm.cdf(-d2_val)-S*norm.cdf(-d1_val)
    return BS_put

def delta(option_type, S, K, r, T, sigma):
    if option_type == "call":
        return norm.cdf(d1(S, K, r, T, sigma))
    elif option_type == "put":
        return norm.cdf(d1(S, K, r, T, sigma))-1

def gamma(S, K, r, T, sigma):
    return norm.cdf(d1(S, K, r, T, sigma))/S*sigma*np.sqrt(T)

def theta(option_type, S, K, r, T, sigma):
    if option_type == "call":
        return -S*norm.cdf(d1(S, K, r, T, sigma))*sigma/2*np.sqrt(T)-r*K*np.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))
    elif option_type == "put":
        return -S*norm.cdf(d1(S, K, r, T, sigma))*sigma/2*np.sqrt(T)+r*K*np.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))

def vega(S, K, r, T, sigma):
    return S*np.sqrt(T)*norm.cdf(d1(S, K, r, T, sigma))

def rho(option_type, S, K, r, T, sigma):
    if option_type == "call":
        return K*T*np.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))
    elif option_type == "put":
        return -K*T*np.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))

st.set_page_config(layout="wide")
st.title("Black-Scholes Option Pricing")
col1, col2 = st.columns(2)

with st.sidebar:
    st.title("📈 Black-Scholes Model")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Arthur Villela`</a>', unsafe_allow_html=True)

    cp = st.sidebar.number_input("Current Asset Price (S)", value=50.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    sp = st.sidebar.number_input("Strike (K)", value=70.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    ty = st.sidebar.number_input("Years to Maturity (T)", value=1.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.4f")
    vol = st.sidebar.number_input("Volatility (σ)", value=0.30, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    rfir = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.15, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

def print_value(S, K, r, T, sigma):
    with col1:
        st.subheader("Call Value")
        st.title(f":blue-background[{round(call_value(S, K, r, T, sigma), 2)}]")

    with col2:
        st.subheader("Put Value")
        st.title(f":green-background[{round(put_value(S, K, r, T, sigma), 2)}]")

print_value(cap, sp, rfir, ty, vol)

st.title("Options Heatmap")

col1, col2 = st.columns(2)

def heat_map(col, row, title):
    plt.figure(figsize=(10,10))
    if title == "Call":
        sn.heatmap(data=data_call, annot=True, fmt=".2f", cmap="flare", xticklabels=col, yticklabels=row, square=True, cbar_kws={"shrink":0.8}, linewidths=.2)
    else:
        sn.heatmap(data=data_put, annot=True, fmt=".2f", cmap="flare", xticklabels=col, yticklabels=row, square=True, cbar_kws={"shrink":0.8}, linewidths=.2)
    plt.xlabel("Asset Price")
    plt.ylabel("volatility")
    st.pyplot(plt)
    plt.close(None)

st.sidebar.write("--------------------------")
st.sidebar.subheader("Heatmap Parameters")
min_vol = st.sidebar.slider("Min volatility", 0.01, 1.00, vol*0.5)
max_vol = st.sidebar.slider("Max Volatility", 0.01, 1.00, vol*1.5)
min_price = st.sidebar.number_input("Min Price", value=cap*0.8, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
max_price = st.sidebar.number_input("Max Price", value=cap*1.2, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

#creating the values to multiply for the heatmap
rows = [(min_vol + i*(max_vol-min_vol)/9) for i in range(0, 10)] #volatility (y-axis)
columns = [(min_price + i*(max_price-min_price)/9) for i in range(0, 10)] #spot price (x-axis)

#printing out the x-axis and y-axis values for the heatmap
rows_print = [round((min_vol + i*(max_vol-min_vol)/9), 2) for i in range(0, 10)]
columns_print = [round((min_price + i*(max_price-min_price)/9), 2) for i in range(0, 10)]

#creating the 2d matrix's for the heat maps
data_call = []
data_put = []
for i in range(len(rows)):
    data_call_row = []
    data_put_row = []
    for j in range(len(columns)):
        call_val = call_value(columns[j], sp, rfir, ty, rows[i])
        put_val = put_value(columns[j], sp, rfir, ty, rows[i])
        data_call_row.append(call_val)
        data_put_row.append(put_val)
    data_call.append(data_call_row)
    data_put.append(data_put_row)

#outputting the heatmaps to the screen
with col1:
    st.header("Call")  
    heat_map(columns_print, rows_print, "Call")

with col2:
    st.header("Put")  
    heat_map(columns_print, rows_print, "Put")

st.title("Greeks")

col1, col2 = st.columns(2)

with col1:
    st.header("Call")  
    st.subheader(f"**Delta:**:blue-background[{round(delta('call',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Gamma:**:blue-background[{round(gamma(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Theta:**:blue-background[{round(theta('call',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Vega:**:blue-background[{round(vega(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Rho:**:blue-background[{round(rho('call', cap, sp, rfir, ty, vol), 3)}]")

with col2:
    st.header("Put")  
    st.subheader(f"**Delta:**:green-background[{round(delta('put',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Gamma:**:green-background[{round(gamma(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Theta:**:green-background[{round(theta('put',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Vega:**:green-background[{round(vega(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Rho:**:green-background[{round(rho('put', cap, sp, rfir, ty, vol), 3)}]")

    