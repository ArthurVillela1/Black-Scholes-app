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

    cap = st.sidebar.number_input("Current Asset Price (S)", value=50.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
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
    plt.figure(figsize=(8, 8))  # Increase the size of the heatmap for better visibility
    if title == "Call":
        sn.heatmap(data=data_call, annot=True, fmt=".2f", cmap="crest", xticklabels=col, yticklabels=row, square=True, cbar_kws={"shrink":0.8})
    else:
        sn.heatmap(data=data_put, annot=True, fmt=".2f", cmap="crest", xticklabels=col, yticklabels=row, square=True, cbar_kws={"shrink":0.8})
    plt.xlabel("Asset Price")
    plt.ylabel("Volatility")

    plt.tight_layout(pad=0)

    st.pyplot(plt)
    plt.close(None)

st.sidebar.write("--------------------------")
st.sidebar.subheader("Heatmap Parameters")
min_vol = st.sidebar.slider("Min volatility", 0.01, 1.00, vol)
max_vol = st.sidebar.slider("Max Volatility", 0.01, 1.00, vol*2)
min_price = round(st.sidebar.number_input("Min Price", value=round(cap*0.8, 2), step=0.01, min_value=0.0, max_value=9999.00, format="%.2f"), 2)
max_price = round(st.sidebar.number_input("Max Price", value=round(cap*1.2, 2), step=0.01, min_value=0.0, max_value=9999.00, format="%.2f"), 2)

# Fixing the number of rows and columns for consistent heatmap dimensions
fixed_rows = 10
fixed_columns = 10

# Creating the values to multiply for the heatmap
rows = np.linspace(min_vol, max_vol, fixed_rows)
columns = np.linspace(min_price, max_price, fixed_columns)

# Round the axis labels to two decimal places for better readability
rows_print = [round(x, 2) for x in rows]
columns_print = [round(x, 2) for x in columns]

# Creating the 2D matrices for the heat maps
data_call = [[call_value(S, sp, rfir, ty, sigma) for S in columns] for sigma in rows]
data_put = [[put_value(S, sp, rfir, ty, sigma) for S in columns] for sigma in rows]

# Outputting the heatmaps to the screen
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
    st.subheader(f"**Delta (∆):** :blue-background[{round(delta('call',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Gamma (Γ):** :blue-background[{round(gamma(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Theta (Θ):** :blue-background[{round(theta('call',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Vega (ν):** :blue-background[{round(vega(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Rho (ρ):** :blue-background[{round(rho('call', cap, sp, rfir, ty, vol), 3)}]")

with col2:
    st.header("Put")  
    st.subheader(f"**Delta (∆):** :green-background[{round(delta('put',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Gamma (Γ):** :green-background[{round(gamma(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Theta (Θ):** :green-background[{round(theta('put',cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Vega (ν):** :green-background[{round(vega(cap, sp, rfir, ty, vol), 3)}]")
    st.subheader(f"**Rho (ρ):** :green-background[{round(rho('put', cap, sp, rfir, ty, vol), 3)}]")