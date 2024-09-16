import streamlit as st
import numpy as np
from scipy.stats import norm
import seaborn as sn
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Calculating d1
def d1(S, K, r, T, sigma):
    return (np.log(S/K)+T*(r+(sigma**2)/2))/(sigma*np.sqrt(T))

# Calculating d2
def d2(S, K, r, T, sigma):
    return d1(S, K, r, T, sigma) - sigma*np.sqrt(T)

# Call value calculation through Black-Scholes formula
def call_value(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    BS_call = S*norm.cdf(d1_val)-K*np.exp(-r*T)*norm.cdf(d2_val)
    return BS_call

# Put value calculation through Black-Scholes formula
def put_value(S, K, r, T, sigma):
    d1_val = d1(S, K, r, T, sigma)
    d2_val = d2(S, K, r, T, sigma)
    BS_put = K*np.exp(-r*T)*norm.cdf(-d2_val)-S*norm.cdf(-d1_val)
    return BS_put

# Greeks calculation
def delta(option_type, S, K, r, T, sigma):
    if option_type == "Call":
        return norm.cdf(d1(S, K, r, T, sigma))
    elif option_type == "Put":
        return norm.cdf(d1(S, K, r, T, sigma))-1

def gamma(S, K, r, T, sigma):
    return norm.pdf(d1(S, K, r, T, sigma))/(S*sigma*np.sqrt(T))

def theta(option_type, S, K, r, T, sigma):
    if option_type == "Call":
        return -S*norm.pdf(d1(S, K, r, T, sigma))*sigma/2*np.sqrt(T)-r*K*np.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))
    elif option_type == "Put":
        return -S*norm.pdf(d1(S, K, r, T, sigma))*sigma/2*np.sqrt(T)+r*K*np.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))

def vega(S, K, r, T, sigma):
    return S*np.sqrt(T)*norm.pdf(d1(S, K, r, T, sigma))

def rho(option_type, S, K, r, T, sigma):
    if option_type == "Call":
        return K*T*np.exp(-r*T)*norm.cdf(d2(S, K, r, T, sigma))
    elif option_type == "Put":
        return -K*T*np.exp(-r*T)*norm.cdf(-d2(S, K, r, T, sigma))

st.set_page_config(layout="wide")
st.title("Black-Scholes Option Pricing")

expander = st.expander("Learn more")
expander.write('''
    The Black-Scholes model is used for pricing options (either puts or calls)
    through five inputs: underlying asset price (S), option strike price (K), time
    to maturity (T), standard deviation of the underlying asset (σ) and the risk-free interest rate(r).
    The formulas for this model are:
''')
expander.image("https://media.licdn.com/dms/image/v2/C5612AQFau1WAkESoAQ/article-inline_image-shrink_400_744/article-inline_image-shrink_400_744/0/1589198759071?e=1729123200&v=beta&t=TUE6sKvtGeai4zJfQ_Nx8i79YUn1mU9BoXRPrSCiZUw")

col1, col2 = st.columns(2)

# Getting B&S inputs from the sidebars
with st.sidebar:
    st.title("📈 Black-Scholes Model")
    st.write("`Created by: Arthur Villela`")
    linkedin_url = "https://www.linkedin.com/in/arthur-villela"
    github_url ="https://github.com/ArthurVillela1"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"><a href="{github_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;"></a>', unsafe_allow_html=True)
    st.sidebar.write("--------------------------")
    option_type = st.sidebar.radio("Option Type", ("Vanilla", "Barrier"))
    if option_type == "Barrier":
        barrier_type = st.sidebar.radio("Barrier Type", ("Knock-In", "Knock-Out"))
    st.sidebar.write("--------------------------")

    s = st.sidebar.number_input("Current Asset Price (S)", value=50.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    k = st.sidebar.number_input("Strike (K)", value=70.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    t = st.sidebar.number_input("Years to Maturity (T)", value=1.00, step=0.01, min_value=0.0, max_value=9999.00, format="%.4f")
    vol = st.sidebar.number_input("Volatility (σ)", value=0.30, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")
    rf = st.sidebar.number_input("Risk-Free Interest Rate (r)", value=0.15, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f")

# Printing call and put values
def fair_values(S, K, r, T, sigma):
    with col1:
        st.header("Call Value")
        st.title(f":blue-background[{round(call_value(S, K, r, T, sigma), 2)}]")

    with col2:
        st.header("Put Value")
        st.title(f":green-background[{round(put_value(S, K, r, T, sigma), 2)}]")

fair_values(s, k, rf, t, vol)

st.title("P&L Heatmap")
col1, col2 = st.columns(2)

# Plotting heatmap
def heat_map(col, row, title):
    plt.figure(figsize=(25, 3)) 
    if title == "Call":
        sn.heatmap(data=data_call, annot=True, fmt=".2f", cmap="RdBu", xticklabels=col, yticklabels=row, square=True, cbar_kws={"shrink": 0.5}, annot_kws={"size": 5})  # Smaller annotation size, bold for clarity
    else:
        sn.heatmap(data=data_put, annot=True, fmt=".2f", cmap="RdBu", xticklabels=col, yticklabels=row, square=True, cbar_kws={"shrink": 0.5}, annot_kws={"size": 5})  # Smaller annotation size, bold for clarity
    plt.xlabel("Spot Price", fontsize=5)
    plt.ylabel("Volatility", fontsize=5)
    plt.xticks(rotation=45, ha="right", fontsize=5)
    plt.yticks(rotation=0, fontsize=5)
    plt.tight_layout(pad=0)
    st.pyplot(plt)
    plt.close(None)

st.sidebar.write("--------------------------")
st.sidebar.subheader("Heatmap Parameters")

with st.sidebar:
    add_radio = st.radio(
        "Option Class",
        ("Call", "Put"),
    )

# Getting the heatmap parameters from the sidebar
purchase_price = round(st.sidebar.number_input("Purchase Price", value=1.0, step=0.01, min_value=0.0, max_value=9999.00, format="%.2f"), 2)
min_vol = st.sidebar.slider("Min Volatility", 0.01, 1.00, vol)
max_vol = st.sidebar.slider("Max Volatility", 0.01, 1.00, vol*2)
min_price = round(st.sidebar.number_input("Min Price", value=round(s*0.5, 2), step=0.01, min_value=0.0, max_value=9999.00, format="%.2f"), 2)
max_price = round(st.sidebar.number_input("Max Price", value=round(s*1.5, 2), step=0.01, min_value=0.0, max_value=9999.00, format="%.2f"), 2)

# Creating evenly spaced numbers between min_vol and max_vol
rows = np.linspace(min_vol, max_vol, 10)
columns = np.linspace(min_price, max_price, 10)

# Rounding numbers for better layout
rows_print = [round(x, 2) for x in rows]
columns_print = [round(x, 2) for x in columns]

# Creating matrices for the heat maps
data_call = [[call_value(S, k, rf, t, sigma)-purchase_price for S in columns] for sigma in rows]
data_put = [[put_value(S, k, rf, t, sigma)- purchase_price for S in columns] for sigma in rows]

with st.container():
    if add_radio == "Call":
        heat_map(columns_print, rows_print, "Call")
    else:
        heat_map(columns_print, rows_print, "Put")

st.title("Greeks")
col1, col2 = st.columns(2)

# Displaying greeks
with col1:
    st.header("Call")  
    st.subheader(f"**Delta (∆):** :blue-background[{round(delta('Call',s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Gamma (Γ):** :blue-background[{round(gamma(s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Theta (Θ):** :blue-background[{round(theta('Call',s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Vega (ν):** :blue-background[{round(vega(s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Rho (ρ):** :blue-background[{round(rho('Call', s, k, rf, t, vol), 2)}]")

with col2:
    st.header("Put")  
    st.subheader(f"**Delta (∆):** :green-background[{round(delta('Put',s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Gamma (Γ):** :green-background[{round(gamma(s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Theta (Θ):** :green-background[{round(theta('Put',s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Vega (ν):** :green-background[{round(vega(s, k, rf, t, vol), 2)}]")
    st.subheader(f"**Rho (ρ):** :green-background[{round(rho('Put', s, k, rf, t, vol), 2)}]")

selected_variable = st.selectbox("", ["Stock Price (S)", "Strike Price (K)", "Time to Maturity (T)", "Volatility (σ)", "Risk-Free Interest Rate (r)"])
col1, col2 = st.columns(2)

# Plotting greeks
def plot_greeks(variable, values, S, K, T, r, sigma, option_type):

    fig = go.Figure()

    if variable == "Stock Price (S)":
        delta_values = [delta(option_type, S, K, r, T, sigma) for S in values]
        gamma_values = [gamma(S, K, r, T, sigma) for S in values]
        theta_values = [theta(option_type, S, K, r, T, sigma) for S in values]
        vega_values = [vega(S, K, r, T, sigma) for S in values]
        rho_values = [rho(option_type, S, K, r, T, sigma) for S in values]
    elif variable == "Strike Price (K)":
        delta_values = [delta(option_type, S, K, r, T, sigma) for K in values]
        gamma_values = [gamma(S, K, r, T, sigma) for K in values]
        theta_values = [theta(option_type, S, K, r, T, sigma) for K in values]
        vega_values = [vega(S, K, r, T, sigma) for K in values]
        rho_values = [rho(option_type, S, K, r, T, sigma) for K in values]
    elif variable == "Time to Maturity (T)":
        delta_values = [delta(option_type, S, K, r, T, sigma) for T in values]
        gamma_values = [gamma(S, K, r, T, sigma) for T in values]
        theta_values = [theta(option_type, S, K, r, T, sigma) for T in values]
        vega_values = [vega(S, K, r, T, sigma) for T in values]
        rho_values = [rho(option_type, S, K, r, T, sigma) for T in values]
    elif variable == "Risk-Free Interest Rate (r)":
        delta_values = [delta(option_type, S, K, r, T, sigma) for r in values]
        gamma_values = [gamma(S, K, r, T, sigma) for r in values]
        theta_values = [theta(option_type, S, K, r, T, sigma) for r in values]
        vega_values = [vega(S, K, r, T, sigma) for r in values]
        rho_values = [rho(option_type, S, K, r, T, sigma) for r in values]
    elif variable == "Volatility (σ)":
        delta_values = [delta(option_type, S, K, r, T, sigma) for sigma in values]
        gamma_values = [gamma(S, K, r, T, sigma) for sigma in values]
        theta_values = [theta(option_type, S, K, r, T, sigma) for sigma in values]
        vega_values = [vega(S, K, r, T, sigma) for sigma in values]
        rho_values = [rho(option_type, S, K, r, T, sigma) for sigma in values]
    
    fig.add_trace(go.Scatter(x=values, y=delta_values, mode='lines', name=f'Delta'))
    fig.add_trace(go.Scatter(x=values, y=gamma_values, mode='lines', name=f'Gamma'))
    fig.add_trace(go.Scatter(x=values, y=theta_values, mode='lines', name=f'Theta'))
    fig.add_trace(go.Scatter(x=values, y=vega_values, mode='lines', name=f'Vega'))
    fig.add_trace(go.Scatter(x=values, y=rho_values, mode='lines', name=f'Rho'))

    fig.update_layout(title=f'{option_type.capitalize()} Option Greeks x {variable}',
                      xaxis_title=variable,
                      yaxis_title='Greek Value')
    return fig

if selected_variable == "Stock Price (S)":
    values = np.linspace(50, 150, 100)
elif selected_variable == "Strike Price (K)":  
    values = np.linspace(50, 150, 100)
elif selected_variable == "Time to Maturity (T)":
    values = np.linspace(0.1, 2, 100)
elif selected_variable == "Risk-Free Interest Rate (r)":
    values = np.linspace(0.0, 0.1, 100)
elif selected_variable == "Volatility (σ)":
    values = np.linspace(0.1, 0.5, 100)

with col1:
    fig_greeks = plot_greeks(selected_variable, values, s, k, t, rf, vol, "Call")
    st.plotly_chart(fig_greeks)

with col2:
    fig_greeks = plot_greeks(selected_variable, values, s, k, t, rf, vol, "Put")
    st.plotly_chart(fig_greeks)