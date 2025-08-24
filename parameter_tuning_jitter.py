# 최적 파라미터 탐색 + jitter
# =========================================================
# 설치(최초 1회)
# =========================================================
# Colab/로컬: 주석 해제 후 1회 실행
!pip install yfinance ta pandas numpy plotly optuna --quiet

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import optuna
from ta.momentum import RSIIndicator
import plotly.graph_objects as go

# =========================================================
# 0) 사용자 입력(선택)
# =========================================================
total_amount = None
try:
    s = input("총 투자 금액을 입력하세요 (숫자만, 미입력 시 비중만 표시): ").strip()
    if s:
        total_amount = float(s)
except Exception:
    print("⚠️ 유효한 숫자가 아닙니다. 비중만 표시합니다.")
    total_amount = None

# =========================================================
# 글로벌 파라미터
# =========================================================
START_DATE = "2016-01-01"

# 현금 수익률 가정 (연 2%) → 일할(365일)
CASH_YIELD_ANNUAL = 0.02
CASH_DAILY = (1.0 + CASH_YIELD_ANNUAL) ** (1.0 / 365.0) - 1.0

# 거래 수수료(사이드당 비율, 예: 0.001 = 0.1% / 매수와 매도 각각 부과)
FEE_RATE_PER_SIDE = 0.001

# 실행 지연(신호 계산일 다음날 체결)
EXECUTION_LAG_DAYS = 1

# 성과지표 연환산 기준(크립토 일봉: 365일)
ANN_FACTOR = 365

# 최적화 목적함수 가중치
OBJ_W_SORTINO = 0.65
OBJ_W_CALMAR  = 0.35

# Start-Date Jitter(강건성 평가용)
ROBUST_OFFSETS_DAYS = [0, 200, 400, 600]

# 최근 의사결정 스케줄 표시 개수
N_DECISION_PRINT = 10

# =========================================================
# 유틸: 1차원 보장 함수(DF(n,1) → Series)
# =========================================================
def to_series_1d(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            raise ValueError("Expected 1 column DataFrame or Series.")
    elif isinstance(x, pd.Series):
        s = x
    else:
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        s = pd.Series(x)
    return pd.Series(s.values, index=s.index, dtype="float64")

# =========================================================
# 1) 데이터 로드
# =========================================================
def load_prices(tickers, start="2016-01-01", interval="1d"):
    frames = []
    for t in tickers:
        df = yf.download(t, start=start, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data for {t}")
        sub = df[["Close"]].rename(columns={"Close": t}).astype(float)
        frames.append(sub)
    prices = pd.concat(frames, axis=1).dropna()
    return prices

TICKERS = ["BTC-USD", "ETH-USD"]
prices_all = load_prices(TICKERS, start=START_DATE)

# =========================================================
# 2) 신호계산 공통 함수 (RSI_SMA, 비율 스케일, 과열도, 120SMA)
# =========================================================
def rsi_sma(series, rsi_win=14, sma_win=14):
    s = to_series_1d(series)
    rsi = RSIIndicator(close=s, window=rsi_win).rsi()
    return to_series_1d(rsi.rolling(sma_win).mean())

def expanding_clip_scale(x: pd.Series, low_pct=0.1, high_pct=99.9):
    x = to_series_1d(x)
    out = []
    vals = x.values

    if len(vals) == 0:
        return pd.Series([], dtype="float64")

    out.append(50.0)

    for i in range(1, len(vals)):
        hist = vals[:i]
        v = vals[i]

        lo = np.nanpercentile(hist, low_pct)
        hi = np.nanpercentile(hist, high_pct)

        vv = np.clip(v, lo, hi)

        mn = np.nanmin(hist)
        mx = np.nanmax(hist)

        scaled = 50.0
        if mx > mn:
            scaled = (vv - mn) / (mx - mn) * 100.0

        out.append(np.clip(scaled, 0, 100))

    return pd.Series(out, index=x.index, dtype="float64")

def build_signals_from_prices(pr_df: pd.DataFrame):
    pr_df = pr_df.dropna().copy()
    btc = to_series_1d(pr_df["BTC-USD"])
    eth = to_series_1d(pr_df["ETH-USD"])
    ret_btc = to_series_1d(btc.pct_change())
    ret_eth = to_series_1d(eth.pct_change())

    btc_rsi_sma14 = rsi_sma(btc, 14, 14)
    eth_rsi_sma14 = rsi_sma(eth, 14, 14)

    sig = pd.DataFrame({
        "BTC_Close": btc,
        "ETH_Close": eth,
        "BTC_RSI_SMA": btc_rsi_sma14,
        "ETH_RSI_SMA": eth_rsi_sma14
    }).dropna()

    raw_ratio = to_series_1d((sig["ETH_RSI_SMA"] / sig["BTC_RSI_SMA"]) * 100)
    sig["RSI_Ratio_Scaled_0_100"] = expanding_clip_scale(raw_ratio)
    sig["Market_Heat"] = to_series_1d((sig["BTC_RSI_SMA"] + sig["ETH_RSI_SMA"]) / 2)
    sig["BTC_SMA120"] = to_series_1d(sig["BTC_Close"].rolling(120).mean())

    signals = sig.dropna().copy()
    return signals, ret_btc, ret_eth

signals_full, ret_btc_full, ret_eth_full = build_signals_from_prices(prices_all)

# =========================================================
# 3) 전략 규칙
# =========================================================
def calc_cash_weight(heat, thr_start, thr_max, max_cash):
    if max_cash <= 0:
        return 0.0
    if heat <= thr_start:
        return 0.0
    if heat >= thr_max:
        return float(max_cash)
    slope = max_cash / (thr_max - thr_start)
    return float(slope * (heat - thr_start))

def next_trading_on_or_after(index: pd.DatetimeIndex, target_day: pd.Timestamp):
    pos = index.searchsorted(target_day, side="left")
    if pos >= len(index):
        return None
    return index[pos]

def build_decision_weights(signals: pd.DataFrame,
                           cash_thr_start: int,
                           cash_thr_max: int,
                           max_cash: int,
                           rebalance_hold_days: int) -> pd.DataFrame:
    idx = signals.index
    decision_w = pd.DataFrame(index=idx, columns=["Cash","BTC","ETH"], dtype="float64")
    decision_w.iloc[:] = np.nan

    in_position = False
    next_reb_date = None
    current_target = np.array([100.0, 0.0, 0.0], dtype="float64")

    for t in idx:
        row = signals.loc[t]
        above = (row["BTC_Close"] > row["BTC_SMA120"])

        if (not in_position) and above:
            mh = row["Market_Heat"]
            cash_w = calc_cash_weight(mh, cash_thr_start, cash_thr_max, max_cash)
            invest = max(0.0, 100.0 - cash_w)
            ratio_scaled = row["RSI_Ratio_Scaled_0_100"]
            btc_w = invest * (ratio_scaled/100.0)
            eth_w = invest * (1.0 - ratio_scaled/100.0)
            current_target = np.array([cash_w, btc_w, eth_w], dtype="float64")

            in_position = True
            target = t + pd.Timedelta(days=rebalance_hold_days)
            next_reb_date = next_trading_on_or_after(idx, target)

        elif in_position and (not above):
            current_target = np.array([100.0, 0.0, 0.0], dtype="float64")
            in_position = False
            next_reb_date = None

        elif in_position and (next_reb_date is not None) and (t >= next_reb_date):
            mh = row["Market_Heat"]
            cash_w = calc_cash_weight(mh, cash_thr_start, cash_thr_max, max_cash)
            invest = max(0.0, 100.0 - cash_w)
            ratio_scaled = row["RSI_Ratio_Scaled_0_100"]
            btc_w = invest * (ratio_scaled/100.0)
            eth_w = invest * (1.0 - ratio_scaled/100.0)
            current_target = np.array([cash_w, btc_w, eth_w], dtype="float64")

            target = t + pd.Timedelta(days=rebalance_hold_days)
            next_reb_date = next_trading_on_or_after(idx, target)

        decision_w.loc[t, ["Cash","BTC","ETH"]] = current_target

    return decision_w

def simulate_portfolio(signals: pd.DataFrame,
                       ret_btc: pd.Series,
                       ret_eth: pd.Series,
                       decision_w: pd.DataFrame,
                       execution_lag_days: int,
                       fee_rate_per_side: float,
                       cash_daily: float) -> pd.DataFrame:
    idx = signals.index
    eff_target = decision_w.shift(execution_lag_days).dropna()
    common_idx = eff_target.index.intersection(ret_btc.index).intersection(ret_eth.index)
    eff_target = eff_target.loc[common_idx]
    rb = to_series_1d(ret_btc.loc[common_idx]).fillna(0.0)
    re = to_series_1d(ret_eth.loc[common_idx]).fillna(0.0)

    actual_w = np.array([100.0, 0.0, 0.0], dtype="float64")

    r_port, fee_series = [], []
    w_cash_list, w_btc_list, w_eth_list = [], [], []
    equity = []
    eq = 1.0
    prev_target = np.array([100.0, 0.0, 0.0], dtype="float64")

    for i, t in enumerate(common_idx):
        target_w = eff_target.loc[t, ["Cash","BTC","ETH"]].values.astype(float)

        is_trade_day = not np.allclose(target_w, prev_target)

        fee_frac = 0.0
        if is_trade_day:
            delta = target_w - actual_w
            traded_notional_pct = np.sum(np.abs(delta)) / 2.0 / 100.0
            fee_frac = fee_rate_per_side * traded_notional_pct
            actual_w = target_w.copy()
            prev_target = target_w.copy()

        day_ret_gross = (actual_w[1]/100.0)*rb.loc[t] + (actual_w[2]/100.0)*re.loc[t] + (actual_w[0]/100.0)*cash_daily
        day_ret_net = day_ret_gross - fee_frac

        eq *= (1.0 + day_ret_net)
        equity.append(eq)
        r_port.append(day_ret_net)
        fee_series.append(fee_frac)

        growth = np.array([1.0 + cash_daily, 1.0 + rb.loc[t], 1.0 + re.loc[t]])
        num = (actual_w/100.0) * growth
        denom = np.sum(num)
        if denom > 1e-9:
            actual_w = (num / denom) * 100.0

        w_cash_list.append(actual_w[0])
        w_btc_list.append(actual_w[1])
        w_eth_list.append(actual_w[2])

    res = pd.DataFrame({
        "Cash_w": to_series_1d(pd.Series(w_cash_list, index=common_idx)),
        "BTC_w":  to_series_1d(pd.Series(w_btc_list, index=common_idx)),
        "ETH_w":  to_series_1d(pd.Series(w_eth_list, index=common_idx)),
        "r_btc":  rb.loc[common_idx],
        "r_eth":  re.loc[common_idx],
        "r_port": to_series_1d(pd.Series(r_port, index=common_idx)),
        "fee":    to_series_1d(pd.Series(fee_series, index=common_idx)),
        "equity": to_series_1d(pd.Series(equity, index=common_idx)),
    }, index=common_idx)

    peak = res["equity"].cummax()
    res["dd"] = res["equity"]/peak - 1.0
    return res

def backtest_with_params(signals: pd.DataFrame,
                         ret_btc: pd.Series,
                         ret_eth: pd.Series,
                         cash_thr_start: int,
                         cash_thr_max: int,
                         max_cash: int,
                         execution_lag_days: int,
                         rebalance_hold_days: int,
                         fee_rate_per_side: float,
                         cash_daily: float):
    decision_w = build_decision_weights(
        signals, cash_thr_start, cash_thr_max, max_cash, rebalance_hold_days
    )
    res = simulate_portfolio(
        signals, ret_btc, ret_eth, decision_w,
        execution_lag_days, fee_rate_per_side, cash_daily
    )
    return res, decision_w

# =========================================================
# 4) 성과지표 & 리포팅
# =========================================================
def perf_stats(port_df: pd.DataFrame):
    r = to_series_1d(port_df["r_port"]).dropna()
    if len(r) < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan,
                "Sortino": np.nan, "MDD": np.nan, "Calmar": np.nan}
    ann = ANN_FACTOR
    eq = to_series_1d(port_df["equity"])
    cagr = (eq.iloc[-1]) ** (ann/len(r)) - 1.0
    vol  = r.std() * np.sqrt(ann)
    sharpe = np.nan if vol==0 else cagr/vol
    downside_r = r[r<0]
    downside_std = downside_r.std() * np.sqrt(ann) if len(downside_r)>1 else 0.0
    sortino = np.nan if downside_std==0 else cagr/downside_std
    mdd = to_series_1d(port_df["dd"]).min()
    calmar = np.nan if mdd==0 else cagr/abs(mdd)
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe,
            "Sortino": sortino, "MDD": mdd, "Calmar": calmar}

def decision_schedule(decision_w: pd.DataFrame,
                      signals_index: pd.DatetimeIndex,
                      lag_days: int) -> pd.DataFrame:
    dw = decision_w.dropna()
    chg = dw[["Cash","BTC","ETH"]].diff().abs().sum(axis=1) > 1e-9
    if len(dw) > 0:
        chg.iloc[0] = True
    dec = dw.loc[chg, ["Cash","BTC","ETH"]].copy()
    exec_dates = []
    for d in dec.index:
        target = d + pd.Timedelta(days=lag_days)
        exec_d = next_trading_on_or_after(signals_index, target)
        exec_dates.append(exec_d)
    dec.insert(0, "DecisionDate", dec.index.date)
    dec.insert(1, "ExecDate", [e.date() if e is not None else None for e in exec_dates])
    return dec

def print_decision_schedule(decision_w: pd.DataFrame,
                            signals_index: pd.DatetimeIndex,
                            lag_days: int,
                            title: str,
                            n_last: int = 10):
    sched = decision_schedule(decision_w, signals_index, lag_days)
    tail = sched.tail(n_last)
    print(f"\n[{title}] 최근 {len(tail)}회 의사결정 스케줄 (결정일 → 체결일, 타깃 비중)")
    print(f"{'결정일':<12} | {'체결일':<12} | {'Cash%':>7} | {'BTC%':>7} | {'ETH%':>7}")
    print("-"*56)
    for _, row in tail.iterrows():
        print(f"{str(row['DecisionDate']):<12} | {str(row['ExecDate']):<12} | "
              f"{row['Cash']:7.2f} | {row['BTC']:7.2f} | {row['ETH']:7.2f}")

def print_portfolio_table(decision_w: pd.DataFrame, total_amount: float | None):
    latest_decision_date = decision_w.dropna().index[-1]
    latest_w = decision_w.loc[latest_decision_date]
    print("\n---")
    print(f"✅ 최신 포트폴리오 (결정일 기준: {latest_decision_date.date()})")
    header = f"{'자산':<12} | {'비중 (%)':>10}"
    if total_amount is not None:
        header += f" | {'예상 금액 ':>18}"
    print(header)
    print("-" * (len(header)+5))
    rows = [("💵 현금", latest_w["Cash"]), ("🟠 비트코인", latest_w["BTC"]), ("🔵 이더리움", latest_w["ETH"])]
    tw = 0.0
    for name, w in rows:
        tw += w
        line = f"{name:<12} | {w:>9.2f}%"
        if total_amount is not None:
            amt = total_amount * (w/100.0)
            line += f" | {amt:18,.0f}"
        print(line)
    print("-" * (len(header)+5))
    footer = f"{'📊 총 합계':<12} | {tw:>9.2f}%"
    if total_amount is not None:
        footer += f" | {total_amount:18,.0f}"
    print(footer)
    print("---")

# =========================================================
# 5) 강건 목적함수: 여러 시작일(오프셋) 평균
# =========================================================
def objective_score_on_slice(prices_slice: pd.DataFrame,
                             cs: int, cm: int, mx: int, rd: int) -> float | None:
    try:
        signals_s, rb_s, re_s = build_signals_from_prices(prices_slice)
        if len(signals_s) < 250:
            return None
        res, _ = backtest_with_params(
            signals_s, rb_s, re_s,
            cash_thr_start=cs, cash_thr_max=cm, max_cash=mx,
            execution_lag_days=EXECUTION_LAG_DAYS,
            rebalance_hold_days=rd,
            fee_rate_per_side=FEE_RATE_PER_SIDE,
            cash_daily=CASH_DAILY
        )
        st = perf_stats(res)
        srt, cal = st["Sortino"], st["Calmar"]
        if pd.isna(srt) or pd.isna(cal) or np.isinf(srt) or np.isinf(cal):
            return -1e9
        return OBJ_W_SORTINO*float(srt) + OBJ_W_CALMAR*float(cal)
    except Exception:
        return None

def robust_objective_score(cs: int, cm: int, mx: int, rd: int,
                           offsets_days=ROBUST_OFFSETS_DAYS) -> float:
    scores = []
    for off in offsets_days:
        pr_slice = prices_all.iloc[off:].copy()
        sc = objective_score_on_slice(pr_slice, cs, cm, mx, rd)
        if sc is not None:
            scores.append(sc)
    return float(np.mean(scores)) if scores else -1e9

def robust_evaluate_params(cs: int, cm: int, mx: int, rd: int,
                           offsets_days=ROBUST_OFFSETS_DAYS) -> pd.DataFrame:
    rows = []
    for off in offsets_days:
        try:
            pr_slice = prices_all.iloc[off:].copy()
            signals_s, rb_s, re_s = build_signals_from_prices(pr_slice)
            if len(signals_s) < 250:
                continue
            res, _ = backtest_with_params(
                signals_s, rb_s, re_s,
                cash_thr_start=cs, cash_thr_max=cm, max_cash=mx,
                execution_lag_days=EXECUTION_LAG_DAYS,
                rebalance_hold_days=rd,
                fee_rate_per_side=FEE_RATE_PER_SIDE,
                cash_daily=CASH_DAILY
            )
            st = perf_stats(res)
            obj = OBJ_W_SORTINO*float(st["Sortino"]) + OBJ_W_CALMAR*float(st["Calmar"])
            rows.append({
                "offset_days": off,
                "Objective": obj,
                "Sortino": float(st["Sortino"]) if pd.notna(st["Sortino"]) else np.nan,
                "Calmar":  float(st["Calmar"])  if pd.notna(st["Calmar"])  else np.nan,
                "CAGR":    float(st["CAGR"])    if pd.notna(st["CAGR"])    else np.nan,
                "MDD":     float(st["MDD"])     if pd.notna(st["MDD"])     else np.nan,
            })
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("offset_days")

# =========================================================
# 6) BASELINE (MAX_CASH=0) & OPTUNA(강건 목적함수) 비교
# =========================================================
BASELINE_PARAMS = {"CASH_START": 55, "CASH_MAX": 100, "MAX_CASH": 0, "REBAL_DAYS": 30}

print("1. Baseline 전략 백테스트 실행 중...")
baseline_res, baseline_decision_w = backtest_with_params(
    signals_full, ret_btc_full, ret_eth_full,
    cash_thr_start=BASELINE_PARAMS["CASH_START"],
    cash_thr_max=BASELINE_PARAMS["CASH_MAX"],
    max_cash=BASELINE_PARAMS["MAX_CASH"],
    execution_lag_days=EXECUTION_LAG_DAYS,
    rebalance_hold_days=BASELINE_PARAMS["REBAL_DAYS"],
    fee_rate_per_side=FEE_RATE_PER_SIDE,
    cash_daily=CASH_DAILY
)
baseline_stats = perf_stats(baseline_res)
print("   Baseline 전략 백테스트 완료.")

print("\n2. Optuna를 이용한 강건 최적화 실행 중...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.TPESampler(seed=23)
study = optuna.create_study(direction="maximize", sampler=sampler)

def objective(trial: optuna.trial.Trial):
    """
    [수정됨] cs=100일 때 cm의 탐색 범위 오류를 해결.
    cm의 하한(low)이 상한(high)보다 커지는 경우, 해당 trial을 즉시 중단(prune)
    """
    cs = trial.suggest_int("CASH_START", 30, 100, step=10)

    cm_low = cs + 10
    if cm_low > 100:
        return -1e9 # 유효하지 않은 파라미터 조합이므로 낮은 점수 반환

    cm = trial.suggest_int("CASH_MAX",   cm_low, 100, step=10)
    mx = trial.suggest_int("MAX_CASH",   0, 100, step=10)
    rd = trial.suggest_int("REBAL_DAYS", 10, 200, step=10)

    score = robust_objective_score(cs, cm, mx, rd, offsets_days=ROBUST_OFFSETS_DAYS)
    return score

study.optimize(objective, n_trials=200, show_progress_bar=True)
print("   Optuna 최적화 완료.")

best = study.best_params
best_score = study.best_value

print("\n3. 최적 파라미터로 최종 백테스트 및 리포트 생성 중...")
opt_res, opt_decision_w = backtest_with_params(
    signals_full, ret_btc_full, ret_eth_full,
    cash_thr_start=best["CASH_START"],
    cash_thr_max=best["CASH_MAX"],
    max_cash=best["MAX_CASH"],
    execution_lag_days=EXECUTION_LAG_DAYS,
    rebalance_hold_days=best["REBAL_DAYS"],
    fee_rate_per_side=FEE_RATE_PER_SIDE,
    cash_daily=CASH_DAILY
)
opt_stats = perf_stats(opt_res)

robust_df = robust_evaluate_params(
    best["CASH_START"], best["CASH_MAX"], best["MAX_CASH"], best["REBAL_DAYS"],
    offsets_days=ROBUST_OFFSETS_DAYS
)
print("   리포트 생성 완료.")

# =========================================================
# 7) 결과 요약 출력 (Baseline vs Optimized + Robustness)
# =========================================================
def fmt_pct(x): return "nan" if pd.isna(x) else f"{x: .2%}"
def fmt4(x):  return "nan" if pd.isna(x) else f"{x: .4f}"

print("\n\n================= 📜 GLOBALS =================")
print(f"Start Date: {START_DATE}")
print(f"Cash Yield (annual): {CASH_YIELD_ANNUAL:.2%}  → daily: {CASH_DAILY:.6f}")
print(f"Fee per side: {FEE_RATE_PER_SIDE:.4%}")
print(f"Execution lag (days): {EXECUTION_LAG_DAYS}")
print(f"Annualization factor: {ANN_FACTOR}")
print(f"Objective: {OBJ_W_SORTINO}*Sortino + {OBJ_W_CALMAR}*Calmar (Start-Date Jitter {ROBUST_OFFSETS_DAYS})")

print("\n================= ⚙️ PARAMS =================")
print(f"BASELINE : {BASELINE_PARAMS}")
print(f"OPTIMIZED: {best} | Best Robust Score(Obj avg): {best_score:.4f}")

print("\n================= 📊 PERFORMANCE COMPARISON =================")
cols = ["CAGR","Vol","Sharpe","Sortino","MDD","Calmar"]
print(f"{'Metric':<10} | {'Baseline':>12} | {'Optimized':>12}")
print("-"*40)
for k in cols:
    b = baseline_stats[k]
    o = opt_stats[k]
    fb = fmt_pct(b) if k in ["CAGR","Vol","MDD"] else fmt4(b)
    fo = fmt_pct(o) if k in ["CAGR","Vol","MDD"] else fmt4(o)
    print(f"{k:<10} | {fb:>12} | {fo:>12}")

print("\n[Baseline - Latest Allocation]")
print_portfolio_table(baseline_decision_w, total_amount)

print("\n[Optimized - Latest Allocation]")
print_portfolio_table(opt_decision_w, total_amount)

print_decision_schedule(baseline_decision_w, signals_full.index, EXECUTION_LAG_DAYS,
                        title="Baseline 결정 스케줄", n_last=N_DECISION_PRINT)
print_decision_schedule(opt_decision_w, signals_full.index, EXECUTION_LAG_DAYS,
                        title="Optimized 결정 스케줄", n_last=N_DECISION_PRINT)

if not robust_df.empty:
    print("\n================= 💪 ROBUSTNESS (Start-Date Jitter) =================")
    print("최적 파라미터의 시작일별 성과 통계:")
    desc = robust_df[["Objective","Sortino","Calmar","CAGR","MDD"]].describe().T
    print(desc.to_string(float_format=lambda x: f"{x: .4f}"))

    figr = go.Figure()
    figr.add_trace(go.Box(y=robust_df["Objective"], name="Objective (across offsets)"))
    figr.update_layout(title="Robustness: Objective Distribution over Start-Date Offsets",
                       yaxis_title="Objective (0.65*Sortino + 0.35*Calmar)")
    figr.show()

    figo = go.Figure()
    figo.add_trace(go.Scatter(x=robust_df["offset_days"], y=robust_df["Objective"],
                              mode="markers+lines", name="Objective"))
    figo.update_layout(title="Objective by Start-Date Offset",
                       xaxis_title="Offset (days)", yaxis_title="Objective")
    figo.show()

# =========================================================
# 8) 시각화: 에쿼티/드로다운 비교, Optuna 이력/산점도
# =========================================================
common_idx = baseline_res.index.intersection(opt_res.index)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=baseline_res.loc[common_idx].index,
                          y=baseline_res.loc[common_idx, "equity"],
                          mode="lines", name="Baseline Equity"))
fig1.add_trace(go.Scatter(x=opt_res.loc[common_idx].index,
                          y=opt_res.loc[common_idx, "equity"],
                          mode="lines", name="Optimized Equity"))
fig1.update_layout(title="Equity Curve: Baseline vs Optimized",
                   xaxis_title="Date", yaxis_title="Equity (Log Scale)", yaxis_type="log")
fig1.show()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=baseline_res.loc[common_idx].index,
                          y=baseline_res.loc[common_idx, "dd"],
                          mode="lines", name="Baseline Drawdown", fill='tozeroy'))
fig2.add_trace(go.Scatter(x=opt_res.loc[common_idx].index,
                          y=opt_res.loc[common_idx, "dd"],
                          mode="lines", name="Optimized Drawdown", fill='tozeroy'))
fig2.update_layout(title="Drawdown: Baseline vs Optimized",
                   xaxis_title="Date", yaxis_title="Drawdown")
fig2.show()

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=study.trials_dataframe()["number"], y=study.trials_dataframe()["value"], mode="lines+markers", name="Objective Value"))
fig3.update_layout(title=f"Optuna Optimization History (Robust Objective)",
                   xaxis_title="Trial #", yaxis_title="Objective Value")
fig3.show()

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=opt_res.index, y=opt_res["BTC_w"], mode="lines", stackgroup="one", name="BTC %", line_color='orange'))
fig4.add_trace(go.Scatter(x=opt_res.index, y=opt_res["ETH_w"], mode="lines", stackgroup="one", name="ETH %", line_color='royalblue'))
fig4.add_trace(go.Scatter(x=opt_res.index, y=opt_res["Cash_w"], mode="lines", stackgroup="one", name="Cash %", line_color='grey'))
fig4.update_layout(title="Optimized Portfolio Weights Over Time",
                   xaxis_title="Date", yaxis_title="Weight (%)", yaxis_range=[0, 100])
fig4.show()
