import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import requests
from scipy.optimize import minimize

st.set_page_config(page_title="GARCH Portfolio Optimizer", layout="wide")

# ----------------------------------------------------------------------
# Local persistence (saved on this computer, in a file next to app.py)
# ----------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "portfolio_data.json"


def load_data():
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_data(data):
    try:
        DATA_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass  # best-effort; don't crash the app if disk write fails


def save_last_session(tickers_list):
    data = load_data()
    data["last_session"] = tickers_list
    save_data(data)


def save_named_portfolio(name, tickers_list):
    data = load_data()
    data.setdefault("saved_portfolios", {})[name] = tickers_list
    save_data(data)


def delete_named_portfolio(name):
    data = load_data()
    data.get("saved_portfolios", {}).pop(name, None)
    save_data(data)


@st.cache_data(ttl=600, show_spinner=False)
def search_symbols(query):
    """Look up matching tickers/companies from Yahoo Finance's search endpoint."""
    query = query.strip()
    if len(query) < 2:
        return []
    try:
        resp = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 8, "newsCount": 0, "listsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5,
        )
        resp.raise_for_status()
        quotes = resp.json().get("quotes", [])
    except Exception:
        return []

    results = []
    for q in quotes:
        symbol = q.get("symbol")
        if not symbol:
            continue
        name = q.get("shortname") or q.get("longname") or ""
        exch = q.get("exchange", "")
        qtype = q.get("quoteType", "")
        results.append({"symbol": symbol, "name": name, "exchange": exch, "type": qtype})
    return results


def resolve_display_name(symbol, known_name):
    """Return the best available full name for a ticker, falling back to a search lookup."""
    if known_name:
        return known_name
    for m in search_symbols(symbol):
        if m["symbol"] == symbol and m["name"]:
            return m["name"]
    return symbol


def render_holdings(weights_series, name_map, bar_color):
    """Render a clean, styled list of holdings with proportional bars."""
    rows = []
    for symbol, weight in weights_series.items():
        name = name_map.get(symbol, symbol)
        pct = weight * 100
        rows.append(f"""
        <div style="margin-bottom:12px;">
            <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:3px;">
                <span style="font-size:0.95em;">
                    <span style="font-weight:600;">{name}</span>
                    <span style="color:#888; font-size:0.82em; margin-left:4px;">{symbol}</span>
                </span>
                <span style="font-weight:600; font-size:0.95em;">{pct:.1f}%</span>
            </div>
            <div style="background:#eee; border-radius:6px; height:8px; width:100%;">
                <div style="background:{bar_color}; width:{pct:.2f}%; height:8px; border-radius:6px;"></div>
            </div>
        </div>
        """)
    st.markdown("".join(rows), unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Covariance shrinkage helper
# ----------------------------------------------------------------------

def pairwise_moments(z_df, min_pair_obs=12):
    """
    Pairwise-complete correlation of each asset's (demeaned, unit-variance) return series.

    Returns
      S        : correlation matrix (unit diagonal; pairs with too little overlap = 0)
      valid    : boolean matrix, True where the pair has >= min_pair_obs overlapping obs
      var_est  : estimated sampling variance of each S_ij  (Var(z_i z_j) / n_ij)
      overlap_txt : short description of the overlaps
    """
    n = z_df.shape[1]
    Z = z_df - z_df.mean()
    Z = Z / Z.std(ddof=0)
    X = Z.to_numpy()
    M = (~np.isnan(X)).astype(float)
    Xf = np.nan_to_num(X, nan=0.0)

    cnt = M.T @ M                                   # overlap n_ij for every pair
    cross = Xf.T @ Xf                               # sum z_i z_j over the overlap
    cross_sq = (Xf ** 2).T @ (Xf ** 2)              # sum z_i^2 z_j^2 over the overlap

    valid = cnt >= min_pair_obs
    np.fill_diagonal(valid, True)
    safe_cnt = np.where(cnt > 0, cnt, 1.0)

    raw = np.where(valid, cross / safe_cnt, 0.0)
    S = np.clip(raw, -1.0, 1.0)
    np.fill_diagonal(S, 1.0)

    var_prod = np.where(valid, cross_sq / safe_cnt - raw ** 2, 0.0)
    var_est = np.clip(var_prod, 0.0, None) / safe_cnt

    off = ~np.eye(n, dtype=bool)
    overlaps = cnt[off]
    overlap_txt = (f"pair overlap min/median/max = {int(overlaps.min())}/"
                   f"{int(np.median(overlaps))}/{int(overlaps.max())}, "
                   f"{int((~valid).sum())} of {n * (n - 1)} pairs below {min_pair_obs} obs set to 0")
    return S, valid, var_est, overlap_txt


def _make_psd(A, min_eig, note):
    """Clip eigenvalues of a symmetric matrix at `min_eig` if needed."""
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    if w.min() < min_eig:
        w = np.clip(w, min_eig, None)
        A = (V * w) @ V.T
        A = (A + A.T) / 2
        return A, note
    return A, ""


def shrunk_covariance(returns_df, vols, min_pair_obs=12, use_shrinkage=True):
    """
    Variance-covariance matrix with optional Ledoit-Wolf shrinkage.

    Steps:
      1. pairwise correlations from all dates where both assets have data
      2. combine with the GARCH long-run volatilities:  S_ij = vol_i * vol_j * corr_ij
      3. shrink the whole matrix (variances and covariances together), if use_shrinkage

      target = mu * I,  mu = average variance   (trace(S) / n)
      Sigma  = (1 - delta) * S + delta * target     (delta = 0 when shrinkage is off)

    delta = min(beta, d) / d, where
      d    = ||S - target||_F^2
      beta = sum_ij vol_i^2 vol_j^2 * Var(corr_ij estimate)
    With complete data and sample volatilities this is exactly sklearn's LedoitWolf.
    Returns (cov, delta, info_string).
    """
    n = returns_df.shape[1]
    S_corr, valid, var_est, overlap_txt = pairwise_moments(returns_df, min_pair_obs)

    cov_S = np.outer(vols, vols) * S_corr
    mu_bar = float(np.trace(cov_S) / n)
    target = mu_bar * np.eye(n)

    w = vols ** 2
    beta = float((np.outer(w, w) * var_est)[valid].sum())
    d = float(((cov_S - target) ** 2).sum())
    delta = float(min(beta, d) / d) if (use_shrinkage and d > 1e-18) else 0.0

    cov = (1 - delta) * cov_S + delta * target
    cov, note = _make_psd(cov, 1e-8 * mu_bar, "matrix repaired to be positive definite")
    if note:
        overlap_txt += "; " + note

    shrink_txt = f"LW shrinkage={delta:.4f}" if use_shrinkage else "no shrinkage"
    info = f"{shrink_txt}; {overlap_txt}"
    return cov, delta, info


# ----------------------------------------------------------------------
# Core optimization logic
# ----------------------------------------------------------------------

def mean_variance_optimization_garch(returns_data, lambda_param, risk_free_rate=0.02, frequency="monthly",
                                      progress_callback=None, min_pair_obs=12, use_shrinkage=True):
    """
    Mean-variance optimization.

    1. Expected return (mu) and long-run volatility of each asset from a Brownian-motion
       model with GARCH(1,1) errors, fitted on that asset's full available history.
    2. Correlations estimated pairwise, using every date where both assets have data.
    3. Variance-covariance matrix = long-run GARCH volatilities + pairwise correlations.
    4. Ledoit-Wolf shrinkage of the whole matrix (optional, use_shrinkage).
    5. Portfolio optimization.
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
        series = log_ret_series.dropna()
        data = series.to_numpy().astype(float)
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

        # --- Long-run (unconditional) variance: sigma_LR^2 = omega / (1 - p - q) ---
        persistence = p_est + q_est
        if persistence < 0.999:
            sigma2_long_run = omega_est / (1 - persistence)
            long_run_ok = True
        else:
            # Persistence too close to/over 1 (near-integrated process): the long-run
            # variance is undefined/explosive, so fall back to the sample variance.
            sigma2_long_run = max(np.var(data) / dt, 1e-8)
            long_run_ok = False

        return {
            "mu": mu_est,
            "sigma2": sigma2_long_run,      # used for optimization (long-run/unconditional variance)
            "sigma2_next": sigma2_next,     # one-step-ahead forecast, kept for reference/logging only
            "omega": omega_est,
            "p": p_est,
            "q": q_est,
            "persistence": persistence,
            "long_run_ok": long_run_ok,
            "n_obs": len(data),
        }

    garch_results = {}
    log_msgs = []
    cols = list(log_returns.columns)
    for i, col in enumerate(cols):
        result = fit_garch_mle(log_returns[col], dt)
        if result is not None:
            garch_results[col] = result
            note = "" if result["long_run_ok"] else "  [persistence≈1, used sample variance instead]"
            log_msgs.append(
                f"{col}: n={result['n_obs']}, mu={result['mu']:.4f}, "
                f"sigma_LR={np.sqrt(result['sigma2']):.4f} "
                f"(1-step={np.sqrt(result['sigma2_next']):.4f}), "
                f"p+q={result['persistence']:.3f}{note}"
            )
        else:
            log_msgs.append(f"{col}: GARCH fit failed (too little data), skipping.")
        if progress_callback:
            progress_callback((i + 1) / len(cols))

    if len(garch_results) < 2:
        raise ValueError("Not enough assets with successful GARCH fits (need at least 2).")

    valid_assets = list(garch_results.keys())

    mu = np.array([garch_results[a]["mu"] for a in valid_assets])
    garch_vols = np.array([np.sqrt(garch_results[a]["sigma2"]) for a in valid_assets])

    # ---- Steps 2-4: pairwise correlations + GARCH vols -> covariance matrix -> shrinkage ----
    cov, delta, cov_info = shrunk_covariance(log_returns[valid_assets], garch_vols,
                                           min_pair_obs=min_pair_obs, use_shrinkage=use_shrinkage)
    log_msgs.append(f"Covariance: {cov_info}")

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
        log_msgs.append("Covariance matrix was not positive definite; repaired.")

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

    asset_names = pd.Index(valid_assets)
    return portfolio, max_sharpe, asset_names, vols, rets, log_msgs


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

st.title("📈 GARCH Mean-Variance Portfolio Optimizer")
st.caption("Fits a GARCH(1,1) model to each asset, then builds an efficient frontier and finds "
           "your optimal portfolio plus the max-Sharpe portfolio.")

with st.sidebar:
    st.header("Settings")

    if "selected_tickers" not in st.session_state:
        # Restore whatever was selected last time the app was run on this computer
        st.session_state.selected_tickers = load_data().get("last_session", [])

    st.subheader("Stocks")
    search_query = st.text_input(
        "🔍 Search by company name or ticker",
        placeholder="e.g. Apple, Microsoft, VWCE",
        key="ticker_search_box",
    )

    if search_query and len(search_query.strip()) >= 2:
        matches = search_symbols(search_query)
        if matches:
            for m in matches:
                already_added = any(t["symbol"] == m["symbol"] for t in st.session_state.selected_tickers)
                c1, c2 = st.columns([5, 1])
                with c1:
                    label = f"**{m['symbol']}** — {m['name']}" if m["name"] else f"**{m['symbol']}**"
                    if m["exchange"]:
                        label += f"  \n*{m['exchange']}*"
                    st.markdown(label)
                with c2:
                    if already_added:
                        st.button("✓", key=f"added_{m['symbol']}", disabled=True)
                    elif st.button("➕", key=f"add_{m['symbol']}"):
                        st.session_state.selected_tickers.append(m)
                        save_last_session(st.session_state.selected_tickers)
                        st.rerun()
        else:
            st.caption("No matches found. You can still add the raw ticker below.")

    with st.expander("Add a raw ticker manually"):
        manual_ticker = st.text_input("Ticker symbol", placeholder="e.g. CSX5.L", key="manual_ticker_box")
        if st.button("Add ticker", key="add_manual_ticker") and manual_ticker.strip():
            symbol = manual_ticker.strip().upper()
            if not any(t["symbol"] == symbol for t in st.session_state.selected_tickers):
                st.session_state.selected_tickers.append({"symbol": symbol, "name": "", "exchange": ""})
                save_last_session(st.session_state.selected_tickers)
                st.rerun()

    st.markdown("**Selected stocks:**")
    if st.session_state.selected_tickers:
        for t in list(st.session_state.selected_tickers):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(f"{t['symbol']}" + (f" — {t['name']}" if t["name"] else ""))
            with c2:
                if st.button("✕", key=f"remove_{t['symbol']}"):
                    st.session_state.selected_tickers = [
                        x for x in st.session_state.selected_tickers if x["symbol"] != t["symbol"]
                    ]
                    save_last_session(st.session_state.selected_tickers)
                    st.rerun()
        if st.button("Clear all", use_container_width=True):
            st.session_state.selected_tickers = []
            save_last_session(st.session_state.selected_tickers)
            st.rerun()
    else:
        st.caption("No stocks selected yet — search above and tap ➕ to add.")

    with st.expander("💾 Saved portfolios"):
        saved = load_data().get("saved_portfolios", {})

        preset_name = st.text_input("Save current list as...", placeholder="e.g. Retirement mix", key="preset_name_box")
        if st.button("Save current portfolio", key="save_preset_btn"):
            if preset_name.strip() and st.session_state.selected_tickers:
                save_named_portfolio(preset_name.strip(), st.session_state.selected_tickers)
                st.success(f"Saved as '{preset_name.strip()}'")
            elif not st.session_state.selected_tickers:
                st.warning("Add some stocks first.")
            else:
                st.warning("Give it a name first.")

        if saved:
            st.divider()
            chosen = st.selectbox("Load a saved portfolio", options=list(saved.keys()), key="load_preset_select")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Load", key="load_preset_btn", use_container_width=True):
                    st.session_state.selected_tickers = saved[chosen]
                    save_last_session(st.session_state.selected_tickers)
                    st.rerun()
            with c2:
                if st.button("Delete", key="delete_preset_btn", use_container_width=True):
                    delete_named_portfolio(chosen)
                    st.rerun()
        else:
            st.caption("No saved portfolios yet.")

    tickers = [t["symbol"] for t in st.session_state.selected_tickers]

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input(
            "Start date",
            value=pd.to_datetime("2015-01-01"),
            min_value=pd.to_datetime("1990-01-01"),
            max_value=pd.to_datetime("today"),
        )
    with col_b:
        end_date = st.date_input(
            "End date",
            value=pd.to_datetime("today"),
            min_value=pd.to_datetime("1990-01-01"),
            max_value=pd.to_datetime("today"),
        )

    frequency = st.selectbox("Data frequency", ["daily", "monthly", "quarterly", "yearly"], index=1)

    risk_free_rate_pct = st.number_input(
        "Risk-free rate (%)",
        value=2.5, step=0.25, format="%.2f",
        help="Annualized risk-free rate as a percentage, e.g. 2.5 for 2.5%"
    )
    risk_free_rate = risk_free_rate_pct / 100.0

    lambda_param = st.number_input(
        "Risk aversion factor (λ) — higher = more conservative",
        value=3.0, step=0.5, format="%.2f"
    )

    use_shrinkage = st.checkbox(
        "Ledoit-Wolf covariance shrinkage",
        value=True,
        help="Shrinks the whole covariance matrix (GARCH variances and pairwise covariances together) "
             "toward a scaled identity: average variance on the diagonal, zero covariances elsewhere."
    )

    min_pair_obs = st.number_input(
        "Min. overlapping observations per pair",
        min_value=3, value=12, step=1,
        help="A correlation between two stocks is estimated on the dates where both have data. "
             "Pairs with fewer overlapping observations than this are treated as uncorrelated. "
             "Use a larger value (e.g. 60) for daily data."
    )

    run_button = st.button("Run Optimization", type="primary", use_container_width=True)

if run_button:
    if len(tickers) < 2:
        st.error("Please add at least 2 stocks using the search box.")
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

    # Non-legacy: a missing price stays NaN instead of being forward-filled into a fake 0% return
    # A missing price stays NaN (no forward-fill into a fake 0% return)
    returns = prices.pct_change(fill_method=None)

    progress_bar = st.progress(0.0, text="Fitting GARCH(1,1) models...")

    def update_progress(frac):
        progress_bar.progress(frac, text=f"Fitting GARCH(1,1) models... {int(frac * 100)}%")

    try:
        portfolio, max_sharpe, asset_names, frontier_vols, frontier_rets, log_msgs = mean_variance_optimization_garch(
            returns, lambda_param=lambda_param, risk_free_rate=risk_free_rate,
            frequency=frequency, progress_callback=update_progress,
            min_pair_obs=int(min_pair_obs), use_shrinkage=use_shrinkage
        )
    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()

    progress_bar.empty()

    with st.expander("GARCH fit log"):
        for msg in log_msgs:
            st.text(msg)

    # Map ticker -> full company name (from search selections, backfilled if needed)
    known_names = {t["symbol"]: t["name"] for t in st.session_state.selected_tickers}
    name_map = {sym: resolve_display_name(sym, known_names.get(sym, "")) for sym in asset_names}

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Optimal Portfolio (λ={lambda_param})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Return", f"{portfolio['return']:.2%}")
        m2.metric("Volatility", f"{portfolio['volatility']:.2%}")
        m3.metric("Sharpe Ratio", f"{portfolio['sharpe']:.3f}")
        weights = pd.Series(portfolio['weights'], index=asset_names, name="Weight")
        weights = weights[weights > 0.0001].sort_values(ascending=False)
        render_holdings(weights, name_map, bar_color="#54A24B")

    with col2:
        st.subheader("Max Sharpe Portfolio")
        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Return", f"{max_sharpe['return']:.2%}")
        m2.metric("Volatility", f"{max_sharpe['volatility']:.2%}")
        m3.metric("Sharpe Ratio", f"{max_sharpe['sharpe']:.3f}")
        weights_sharpe = pd.Series(max_sharpe['weights'], index=asset_names, name="Weight")
        weights_sharpe = weights_sharpe[weights_sharpe > 0.0001].sort_values(ascending=False)
        render_holdings(weights_sharpe, name_map, bar_color="#E45756")

    st.divider()
    st.subheader("Efficient Frontier")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets, mode='lines',
        name='Efficient Frontier',
        line=dict(color='#4C78A8', width=3, shape='spline'),
        hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[portfolio['volatility']], y=[portfolio['return']], mode='markers',
        name=f"Your Portfolio (λ={lambda_param})",
        marker=dict(color='#54A24B', size=16, symbol='circle', line=dict(width=2, color='white')),
        hovertemplate=(f"Your Portfolio<br>Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}"
                        f"<br>Sharpe: {portfolio['sharpe']:.2f}<extra></extra>"),
    ))
    fig.add_trace(go.Scatter(
        x=[max_sharpe['volatility']], y=[max_sharpe['return']], mode='markers',
        name='Max Sharpe',
        marker=dict(color='#E45756', size=20, symbol='star', line=dict(width=2, color='white')),
        hovertemplate=(f"Max Sharpe<br>Volatility: %{{x:.2%}}<br>Return: %{{y:.2%}}"
                        f"<br>Sharpe: {max_sharpe['sharpe']:.2f}<extra></extra>"),
    ))
    fig.update_layout(
        title=f"GARCH(1,1) Mean-Variance Frontier ({frequency.capitalize()} Data)",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="Expected Return (Annualized)",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Search for stocks and set your parameters in the sidebar, then click **Run Optimization**.")
