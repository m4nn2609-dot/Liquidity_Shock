from yfinance import Ticker

ticker=Ticker(ticker="^NSEI")
df=ticker.history(period="5y")

df = df[df['Volume'] > 0]
