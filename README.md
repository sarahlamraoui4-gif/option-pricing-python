# option-pricing-python
Pricing European options with Black-Scholes and Monte Carlo in Python. Includes Greeks and no-arbitrage validation.
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def _check_params(S, K, T, r, sigma):
    if T <= 0:
        raise ValueError(f"T doit être > 0 (reçu : {T})")
    if sigma <= 0:
        raise ValueError(f"sigma doit être > 0 (reçu : {sigma})")
    if S <= 0 or K <= 0:
        raise ValueError(f"S et K doivent être > 0 (reçu : S={S}, K={K})")


def black_scholes(S, K, T, r, sigma, option="call"):
    _check_params(S, K, T, r, sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option doit être 'call' ou 'put'")


def check_parity(S, K, T, r, sigma):

    C = black_scholes(S, K, T, r, sigma, "call")
    P = black_scholes(S, K, T, r, sigma, "put")
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    print(f"\nParité call-put : C-P = {lhs:.4f} | S-Ke^(-rT) = {rhs:.4f} | écart = {abs(lhs-rhs):.6f}")


def monte_carlo(S, K, T, r, sigma, option="call", n=100_000):
    _check_params(S, K, T, r, sigma)

    z = np.random.randn(n)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    if option == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option doit être 'call' ou 'put'")

    return np.exp(-r * T) * payoffs.mean()

def greeks(S, K, T, r, sigma):
    _check_params(S, K, T, r, sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T)
    theta = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, vega, theta, rho

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20

    call_bs = black_scholes(S, K, T, r, sigma, "call")
    put_bs  = black_scholes(S, K, T, r, sigma, "put")
    call_mc = monte_carlo(S, K, T, r, sigma, "call")
    put_mc  = monte_carlo(S, K, T, r, sigma, "put")

    print("=" * 40)
    print(f"{'Option':<10} {'Black-Scholes':>15} {'Monte Carlo':>13}")
    print("-" * 40)
    print(f"{'Call':<10} {call_bs:>15.4f} {call_mc:>13.4f}")
    print(f"{'Put':<10} {put_bs:>15.4f} {put_mc:>13.4f}")
    print("=" * 40)

    check_parity(S, K, T, r, sigma)

    d, g, v, th, rh = greeks(S, K, T, r, sigma)
    print(f"\nGreeks (call) :")
    print(f"  Delta : {d:.4f}")
    print(f"  Gamma : {g:.4f}")
    print(f"  Vega  : {v:.4f}")
    print(f"  Theta : {th:.4f}")
    print(f"  Rho   : {rh:.4f}")

    spots = np.linspace(60, 140, 200)
    call_prices = [black_scholes(s, K, T, r, sigma, "call") for s in spots]
    put_prices  = [black_scholes(s, K, T, r, sigma, "put")  for s in spots]

    plt.figure(figsize=(9, 5))
    plt.plot(spots, call_prices, label="Call", color="steelblue", linewidth=2)
    plt.plot(spots, put_prices,  label="Put",  color="coral",     linewidth=2)
    plt.axvline(K, color="gray", linestyle="--", linewidth=1, label="Strike K=100")
    plt.title("Prix Call & Put selon le spot")
    plt.xlabel("Spot (S)")
    plt.ylabel("Prix")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
