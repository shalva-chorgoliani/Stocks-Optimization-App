import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="GARCH Portfolio Optimizer", layout="wide")

# ----------------------------------------------------------------------
# Core optimization logic (adapted from the original script)
# ----------------------------------------------------------------------

def mean_variance_optimization_garch(returns_data, lambda_param, risk_free_rate=0.02, frequency="monthly",
                                      progress_callback=None):
    """
    Mean-variance optimization using GARCH(1,1)-estimated expected returns and covariances.
    """
    freq_map = {"daily": 252, "monthly": 12, "quarterly": 4, "yearly": 1}
    if frequency not in freq_map:
        raise ValueError(f"Invalid frequency '{frequency}'. Choose from {list(freq_map.keys())}.")
    scale = freq_map[frequency]
    dt = 1.0 / scale

    returns_data = returns_data.select_dtypes(include=[np.number])
    returns_clean = returns_data.dropna(axis=1, how='all')
    log_returns = np.log1p(returns_clean)

    def neg_loglik(params, data, dt):
        mu, omega, p, q = params
        n = len(data)
        if omega <= 0 or p < 0 or q < 0 or p >= 1 or q >= 1:
            return 1e12
        sigma2 = np.empty(n)
        sample_var = np.nanvar(data)
        sigma2[0] = max(sample_var / dt, 1e-8)
        ll = 0.0
        for t in range(n):
            if t > 0:
                resid_prev = data[t - 1] - mu * dt
                sigma2[t] = omega + p * sigma2[t - 1] + q * (resid_prev ** 2) / dt
                if sigma2[t] <= 0 or not np.isfinite(sigma2[t]):
                    return 1e12
            resid = data[t] - mu * dt
            denom = sigma2[t] * dt
            if denom <= 0 or not np.isfinite(denom):
                return 1e12
            ll += 0.5 * (np.log(2 * np.pi) + np.log(denom) + (resid ** 2) / denom)
        return ll

    def fit_garch_mle(log_ret_series, dt):
        data = np.asarray(log_ret_series).astype(float)
        data = data[~np.isnan(data)]
        if len(data) < 10:
            return None
        mu0 = np.mean(data) / dt
        var0 = np.var(data)
        x0 = np.array([mu0, 0.01 * var0, 0.85, 0.10])
        bnds = [(None, None), (1e-12, None), (0.0, 0.999), (0.0, 0.999)]
        res = minimize(lambda x: neg_loglik(x, data, dt), x0,
                        method="L-BFGS-B", bounds=bnds,
                        options={"disp": False, "maxiter": 10000})
        if not res.success:
            return None
        mu_est, omega_est, p_est, q_est = res.x

        n = len(data)
        sigma2 = np.empty(n)
        sigma2[0] = max(np.var(data) / dt, 1e-8)
        for t in range(1, n):
            resid_prev = data[t - 1] - mu_est * dt
            sigma2[t] = omega_est + p_est * sigma2[t - 1] + q_est * (resid_prev ** 2) / dt
        resid_last = data[-1] - mu_est * dt
        sigma2_next = omega_est + p_est * sigma2[-1] + q_est * (resid_last ** 2) / dt

        return {"mu": mu_est, "sigma2": sigma2_next, "omega": omega_est, "p": p_est, "q": q_est}

    garch_results = {}
    log_msgs = []
    cols = list(log_returns.columns)
    for i, col in enumerate(cols):
        result = fit_garch_mle(log_returns[col].dropna().values, dt)
        if result is not None:
            garch_results[col] = result
            log_msgs.append(f"{col}: mu={result['mu']:.4f}, sigma={np.sqrt(result['sigma2']):.4f}")
        else:
            log_msgs.append(f"{col}: GARCH fit failed, skipping.")
        if progress_callback:
            progress_callback((i + 1) / len(cols))

    if len(garch_results) < 2:
        raise ValueError("Not enough assets with successful GARCH fits (need at least 2).")

    valid_assets = list(garch_results.keys())
    log_returns_valid = log_returns[valid_assets]

    mu = np.array([garch_results[a]["mu"] for a in valid_assets])
    garch_vols = np.array([np.sqrt(garch_results[a]["sigma2"]) for a in valid_assets])

    log_ret_array = log_returns_valid.dropna(how='all').to_numpy()
    corr_matrix = np.corrcoef(log_ret_array.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    D = np.diag(garch_vols)
    cov = D @ corr_matrix @ D

    def nearest_positive_definite(A):
        B = (A + A.T) / 2
        _, s, Vt = np.linalg.svd(B)
        H = (Vt.T * s) @ Vt
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        for k in range(11):
            try:
                np.linalg.cholesky(A3)
                return A3
            except np.linalg.LinAlgError:
                mineig = np.min(np.real(np.linalg.eigvals(A3)))
                A3 += np.eye(A3.shape[0]) * (-mineig * 1.01 + 1e-8)
        raise ValueError("Could not fix covariance matrix.")

    eigs = np.linalg.eigvalsh(cov)
    if np.any(eigs <= 0):
        cov = nearest_positive_definite(cov)

    n_assets = len(valid_assets)

    def portfolio_stats(w):
        r = np.dot(w, mu)
        v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        s = (r - risk_free_rate) / v if v > 0 else 0
        return r, v, s

    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bnds = tuple((0, 1) for _ in range(n_assets))
    w0 = np.ones(n_assets) / n_assets

    def utility_neg(w):
        r, v, _ = portfolio_stats(w)
        return -(r - lambda_param * v ** 2)

    res = minimize(utility_neg, w0, method='SLSQP', bounds=bnds, constraints=cons)
    if not res.success:
        raise ValueError(f"Optimization failed: {res.message}")
    r, v, s = portfolio_stats(res.x)
    portfolio = {'lambda': lambda_param, 'return': r, 'volatility': v, 'sharpe': s, 'weights': res.x}

    res2 = minimize(lambda w: -portfolio_stats(w)[2], w0, method='SLSQP', bounds=bnds, constraints=cons)
    r2, v2, s2 = portfolio_stats(res2.x)
    max_sharpe = {'return': r2, 'volatility': v2, 'sharpe': s2, 'weights': res2.x}

    lambda_range = np.logspace(-2, 2, 30)
    vols, rets = [], []
    w_prev = w0.copy()
    for lam in lambda_range:
        def u_neg(w):
            return -(np.dot(w, mu) - 0.5 * lam * np.dot(w.T, np.dot(cov, w)))
        opt = minimize(u_neg, w_prev, method='SLSQP', bounds=bnds, constraints=cons)
        if opt.success:
            r_, v_, _ = portfolio_stats(opt.x)
            rets.append(r_)
            vols.append(v_)
            w_prev = opt.x
        else:
            rets.append(np.nan)
            vols.append(np.nan)

    rets = np.array(rets)
    vols = np.array(vols)
    mask = ~np.isnan(rets)
    rets, vols = rets[mask], vols[mask]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(vols, rets, 'b-', lw=2, label='Efficient Frontier (GARCH)')
    ax.scatter(v, r, color='green', s=100, label=f'Your Portfolio (λ={lambda_param})')
    ax.scatter(v2, r2, color='red', s=150, marker='*', label='Max Sharpe')
    ax.set_xlabel('Volatility (Annualized, GARCH)')
    ax.set_ylabel('Expected Return (Annualized, GARCH)')
    ax.set_title(f'GARCH(1,1) Mean-Variance Frontier ({frequency.capitalize()} Data)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    asset_names = pd.Index(valid_assets)
    return portfolio, max_sharpe, asset_names, fig, log_msgs


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.title("📈 GARCH Mean-Variance Portfolio Optimizer")
st.caption("Fits a GARCH(1,1) model to each asset, then builds an efficient frontier and finds "
           "your optimal portfolio plus the max-Sharpe portfolio.")

DEFAULT_TICKERS = ['CSX5.L', 'EIMI.MI', 'IWQU.MI', 'ZPRV.DE', 'SPYL.DE', 'UETW.DE']

with st.sidebar:
    st.header("Settings")

    tickers_text = st.text_area(
        "Tickers (comma-separated)",
        value=", ".join(DEFAULT_TICKERS),
        help="Use Yahoo Finance ticker symbols, e.g. AAPL, MSFT, VWCE.DE"
    )
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start date", value=pd.to_datetime("2015-01-01"))
    with col_b:
        end_date = st.date_input("End date", value=pd.to_datetime("today"))

    frequency = st.selectbox("Data frequency", ["daily", "monthly", "quarterly", "yearly"], index=1)

    risk_free_rate = st.number_input(
        "Risk-free rate (annualized, e.g. 0.025 = 2.5%)",
        value=0.025, step=0.005, format="%.4f"
    )

    lambda_param = st.number_input(
        "Risk aversion factor (λ) — higher = more conservative",
        value=3.0, step=0.5, format="%.2f"
    )

    run_button = st.button("Run Optimization", type="primary", use_container_width=True)

if run_button:
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
        st.stop()

    with st.spinner(f"Downloading {frequency} price data for {len(tickers)} tickers..."):
        interval_map = {"daily": "1d", "monthly": "1mo", "quarterly": "3mo", "yearly": "1y"}
        try:
            data = yf.download(
                tickers=tickers,
                start=str(start_date),
                end=str(end_date),
                interval=interval_map[frequency],
                group_by='ticker',
                auto_adjust=True,
                progress=False
            )
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    if data.empty:
        st.error("No data returned. Check your tickers and date range.")
        st.stop()

    try:
        if len(tickers) == 1:
            prices = data[['Close']].rename(columns={'Close': tickers[0]})
        else:
            prices = pd.concat(
                {t: data[t]['Close'] for t in tickers if t in data.columns.get_level_values(0) and 'Close' in data[t]},
                axis=1
            )
    except Exception as e:
        st.error(f"Could not parse downloaded data: {e}")
        st.stop()

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        st.warning(f"No data found for: {', '.join(missing)} (skipped)")

    returns = prices.pct_change()

    progress_bar = st.progress(0.0, text="Fitting GARCH(1,1) models...")

    def update_progress(frac):
        progress_bar.progress(frac, text=f"Fitting GARCH(1,1) models... {int(frac * 100)}%")

    try:
        portfolio, max_sharpe, asset_names, fig, log_msgs = mean_variance_optimization_garch(
            returns, lambda_param=lambda_param, risk_free_rate=risk_free_rate,
            frequency=frequency, progress_callback=update_progress
        )
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()

    progress_bar.empty()

    with st.expander("GARCH fit log"):
        for msg in log_msgs:
            st.text(msg)

    st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Optimal Portfolio (λ={lambda_param})")
        st.metric("Expected Return", f"{portfolio['return']:.2%}")
        st.metric("Volatility", f"{portfolio['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{portfolio['sharpe']:.3f}")
        weights = pd.Series(portfolio['weights'], index=asset_names, name="Weight")
        weights = weights[weights > 0.0001].sort_values(ascending=False)
        st.dataframe(weights.apply(lambda x: f"{x:.2%}"), use_container_width=True)

    with col2:
        st.subheader("Max Sharpe Portfolio")
        st.metric("Expected Return", f"{max_sharpe['return']:.2%}")
        st.metric("Volatility", f"{max_sharpe['volatility']:.2%}")
        st.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.3f}")
        weights_sharpe = pd.Series(max_sharpe['weights'], index=asset_names, name="Weight")
        weights_sharpe = weights_sharpe[weights_sharpe > 0.0001].sort_values(ascending=False)
        st.dataframe(weights_sharpe.apply(lambda x: f"{x:.2%}"), use_container_width=True)

else:
    st.info("Set your tickers and parameters in the sidebar, then click **Run Optimization**.")
