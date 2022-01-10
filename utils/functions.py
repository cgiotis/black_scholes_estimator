import numpy as np
from scipy.stats import norm as norm

"""
Option-related parameters:

C = call option price
P = put option price
S = current stock price
K = option strike price
r = risk-free interest rate
T = option expiration date
tau = time to maturity of option (T-now)/360
"""

def find_d1(S, K, r, tau, sigma):
    return (1 / (sigma * np.sqrt(tau))) * (np.log(S / K) + (r + (np.square(sigma) / 2)) * tau)

def find_d2(S, K, r, tau, sigma):
    return find_d1(S, K, r, tau, sigma) - (sigma * np.sqrt(tau))

def black_scholes_call(S, K, r, tau, sigma):
    d1 = find_d1(S, K, r, tau, sigma)
    d2 = find_d2(S, K, r, tau, sigma)
    return (norm.cdf(d1) * S) - (norm.cdf(d2) * K * np.exp(-r * tau))

def black_scholes_put(S, K, r, tau, sigma):
    # derived from put-call parity principle
    d1 = find_d1(S, K, r, tau, sigma)
    d2 = find_d2(S, K, r, tau, sigma)
    return norm.cdf(-d2) * K * np.exp(-r * tau) - norm.cdf(-d1) * S

# Greeks - call
def delta_call(S, K, r, tau, sigma):
    return norm.cdf(find_d1(S, K, r, tau, sigma))

def gamma_call(S, K, r, tau, sigma):
    return norm.pdf(find_d1(S, K, r, tau, sigma)) / (S * (sigma * np.sqrt(tau)))

def vega_call(S, K, r, tau, sigma):
    return S * norm.pdf(find_d1(S, K, r, tau, sigma)) * np.sqrt(tau) * K

def theta_call(S, K, r, tau, sigma):
    return - (S * norm.pdf(find_d1(S, K, r, tau, sigma)) * sigma) / (2 * np.sqrt(tau)) \
            - (r * np.asarray(K) * np.exp(- r * tau) * norm.cdf(find_d2(S, K, r, tau, sigma)))

def rho_call(S, K, r, tau, sigma):
    return np.asarray(K) * tau * np.exp(-r * tau) * norm.cdf(find_d2(S, K, r, tau, sigma))

# Greeks - put
def delta_put(S, K, r, tau, sigma):
    return norm.cdf(find_d1(S, K, r, tau, sigma)) - 1

def gamma_put(S, K, r, tau, sigma):
    return norm.pdf(find_d1(S, K, r, tau, sigma)) / (S * (sigma * np.sqrt(tau)))

def vega_put(S, K, r, tau, sigma):
    return S * norm.pdf(find_d1(S, K, r, tau, sigma)) * np.sqrt(tau)

def theta_put(S, K, r, tau, sigma):
    return - (S * norm.pdf(find_d1(S, K, r, tau, sigma)) * sigma) / (2 * np.sqrt(tau)) \
            + (r * np.asarray(K) * np.exp(- r * tau) * norm.cdf(-find_d2(S, K, r, tau, sigma)))

def rho_put(S, K, r, tau, sigma):
    return - np.asarray(K) * tau * np.exp(-r * tau) * norm.cdf(-find_d2(S, K, r, tau, sigma))

def greeks_call(S, K, r, tau, sigma):
    delta = delta_call(S, K, r, tau, sigma)
    gamma = gamma_call(S, K, r, tau, sigma)
    vega = vega_call(S, K, r, tau, sigma)
    theta = theta_call(S, K, r, tau, sigma)
    rho = rho_call(S, K, r, tau, sigma)

    return (delta, gamma, vega, theta, rho)

def greeks_put(S, K, r, tau, sigma):
    delta = delta_put(S, K, r, tau, sigma)
    gamma = gamma_put(S, K, r, tau, sigma)
    vega = vega_put(S, K, r, tau, sigma)
    theta = theta_put(S, K, r, tau, sigma)
    rho = rho_put(S, K, r, tau, sigma)

    return (delta, gamma, vega, theta, rho)
