
!pip install yfinance ta pandas plotly

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
import plotly.graph_objects as go
from datetime import timedelta

# ==============================================================================
# 0. 총 투자 금액 입력 받기
# ==============================================================================
total_amount = None
try:
    # 사용자에게 총 투자액 입력을 요청합니다.
    total_amount_str = input("총 투자 금액을 입력하세요 (숫자만 입력, 미입력 시 비중만 표시): ")
    if total_amount_str.strip(): # 입력값이 공백이 아닌 경우
        total_amount = float(total_amount_str)
except ValueError:
    print("⚠️ 유효한 숫자를 입력하지 않았습니다. 포트폴리오 비중만 표시합니다.")
    total_amount = None


# ==============================================================================
# 데이터 처리 및 지표 계산 (기존 코드와 동일)
# ==============================================================================
# 1) 데이터 다운로드
btc = yf.download('BTC-USD', start='2018-01-01', interval='1d', progress=False)
eth = yf.download('ETH-USD', start='2018-01-01', interval='1d', progress=False)

# 2) 종가 Series 정렬 & 동일 인덱스 맞추기
btc_close = btc['Close'].squeeze()
eth_close = eth['Close'].squeeze().reindex(btc_close.index, method='nearest')

# 3) RSI & 14-day SMA 계산
btc_rsi = RSIIndicator(close=btc_close, window=14).rsi()
eth_rsi = RSIIndicator(close=eth_close, window=14).rsi()
btc_rsi_sma14 = btc_rsi.rolling(window=14).mean()
eth_rsi_sma14 = eth_rsi.rolling(window=14).mean()

# 4) RSI SMA 데이터프레임 생성
rsidata = pd.DataFrame({
    'BTC_RSI_SMA': btc_rsi_sma14,
    'ETH_RSI_SMA': eth_rsi_sma14
}).dropna()

# 5) SMA 비율 계산 및 0-100 스케일링
ratio = rsidata['ETH_RSI_SMA'] / rsidata['BTC_RSI_SMA'] * 100
trimmed_ratio = ratio.clip(lower=ratio.quantile(0.001), upper=ratio.quantile(0.999))
scaled_ratio = (trimmed_ratio - trimmed_ratio.min()) / (trimmed_ratio.max() - trimmed_ratio.min()) * 100
rsidata['RSI_Ratio_Scaled_0_100'] = scaled_ratio.clip(0, 100)

# 5-1) 시장 과열 지표 및 현금 비중 파라미터 설정
rsidata['Market_Heat'] = (rsidata['BTC_RSI_SMA'] + rsidata['ETH_RSI_SMA']) / 2
CASH_THRESHOLD_START = 60
CASH_THRESHOLD_MAX = 80
MAX_CASH_WEIGHT = 100

# 5-2) 시장 과열 정도에 따른 현금 비중 계산 함수
def calculate_cash_weight(heat_indicator):
    if heat_indicator <= CASH_THRESHOLD_START:
        return 0
    elif heat_indicator >= CASH_THRESHOLD_MAX:
        return MAX_CASH_WEIGHT
    else:
        slope = MAX_CASH_WEIGHT / (CASH_THRESHOLD_MAX - CASH_THRESHOLD_START)
        return slope * (heat_indicator - CASH_THRESHOLD_START)

# 5-3) 현금 비중 및 자산별 최종 비중 계산
rsidata['Cash_Weight'] = rsidata['Market_Heat'].apply(calculate_cash_weight)
rsidata['Invest_Weight'] = 100 - rsidata['Cash_Weight']
rsidata['BTC_Weight'] = rsidata['Invest_Weight'] * (rsidata['RSI_Ratio_Scaled_0_100'] / 100)
rsidata['ETH_Weight'] = rsidata['Invest_Weight'] * ((100 - rsidata['RSI_Ratio_Scaled_0_100']) / 100)


# ==============================================================================
# 6. 포트폴리오 결과 출력 (★ 수정된 부분 ★)
# ==============================================================================
latest_date = rsidata.index[-1].strftime('%Y-%m-%d')
latest_data = rsidata.iloc[-1]

# 표시할 포트폴리오 항목 리스트
portfolio_items = [
    ("💵 현금", latest_data['Cash_Weight']),
    ("🟠 비트코인", latest_data['BTC_Weight']),
    ("🔵 알트코인", latest_data['ETH_Weight'])
]

# 결과 출력
print("\n---")
print(f"✅ 최신 포트폴리오 ({latest_date} 기준)")
print(f"📈 시장 과열 지표 (BTC/ETH RSI SMA 평균): {latest_data['Market_Heat']:.2f}")
print("---")

# 테이블 헤더 생성
header = f"{'자산':<12} | {'비중 (%)':>10}"
if total_amount is not None:
    header += f" | {'예상 금액 ':>18}" #
print(header)
print("-" * (len(header) + 5)) # 구분선 길이 조정

# 테이블 내용 출력
total_weight = 0
for asset, weight in portfolio_items:
    total_weight += weight
    row = f"{asset:<12} | {weight:>9.2f}%"
    if total_amount is not None:
        amount = total_amount * (weight / 100)
        # 금액 포맷팅: 천 단위 쉼표 추가
        row += f" | {amount:18,.0f}"
    print(row)

# 테이블 푸터(합계) 출력
print("-" * (len(header) + 5))
footer = f"{'📊 총 합계':<12} | {total_weight:>9.2f}%"
if total_amount is not None:
    footer += f" | {total_amount:18,.0f}"
print(footer)
print("---\n")


# ==============================================================================
# 차트 생성 (기존 코드와 동일)
# ==============================================================================
# 7) 과매수 마스크
btc_over = rsidata['BTC_RSI_SMA'] > 70
eth_over = rsidata['ETH_RSI_SMA'] > 70
both_over = btc_over & eth_over
btc_only_over = btc_over & ~eth_over
eth_only_over = eth_over & ~btc_over

# 8) 연속 구간 탐색 함수
def get_segments(mask: pd.Series):
    segments, in_seg = [], False
    prev_date = None
    for date, val in mask.items():
        if val and not in_seg:
            start = date; in_seg = True
        if not val and in_seg:
            segments.append((start, prev_date))
            in_seg = False
        prev_date = date
    if in_seg:
        segments.append((start, mask.index[-1]))
    return segments

# 9) 하이라이트용 shapes 생성
shapes = []
for start, end in get_segments(btc_only_over):
    shapes.append(dict(type='rect', xref='x', yref='paper', x0=start, x1=end + timedelta(days=1), y0=0, y1=1, fillcolor='rgba(255, 215, 0, 0.3)', line_width=0, layer='below'))
for start, end in get_segments(eth_only_over):
    shapes.append(dict(type='rect', xref='x', yref='paper', x0=start, x1=end + timedelta(days=1), y0=0, y1=1, fillcolor='rgba(60, 179, 113, 0.3)', line_width=0, layer='below'))
for start, end in get_segments(both_over):
    shapes.append(dict(type='rect', xref='x', yref='paper', x0=start, x1=end + timedelta(days=1), y0=0, y1=1, fillcolor='rgba(255, 99, 71, 0.3)', line_width=0, layer='below'))

# 10) Plotly 인터랙티브 차트
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rsidata.index,
    y=rsidata['RSI_Ratio_Scaled_0_100'],
    mode='lines',
    name='ETH/BTC RSI SMA Ratio (0–100)',
    line=dict(color='black'),
    customdata=rsidata[['BTC_RSI_SMA', 'ETH_RSI_SMA', 'Cash_Weight', 'BTC_Weight', 'ETH_Weight', 'Market_Heat']].values,
    hovertemplate=(
        "<b>%{x|%Y-%m-%d}</b>" +
        "<br>" +
        "<b>Allocation Basis</b><br>" +
        "Scaled Ratio: %{y:.2f}<br>" +
        "Market Heat: %{customdata[5]:.2f}<br>" +
        "BTC RSI SMA: %{customdata[0]:.2f}<br>" +
        "ETH RSI SMA: %{customdata[1]:.2f}" +
        "<br>" +
        "<b>Final Portfolio Weights</b><br>" +
        "💵 Cash: %{customdata[2]:.2f}%<br>" +
        "🟠 BTC: %{customdata[3]:.2f}%<br>" +
        "🔵 ETH: %{customdata[4]:.2f}%" +
        "<extra></extra>"
    )
))

# 범례용 더미 트레이스
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(255, 215, 0, 0.3)', width=10), name='BTC RSI_SMA > 70 Only'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(60, 179, 113, 0.3)', width=10), name='ETH RSI_SMA > 70 Only'))
fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='rgba(255, 99, 71, 0.3)', width=10), name='Both RSI_SMA > 70'))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Scaled Ratio (Basis for BTC/ETH Allocation)',
    shapes=shapes,
    legend=dict(orientation='h', y=1.2, x=0.5, xanchor='center'),
    hovermode='x unified'
)

fig.show()
