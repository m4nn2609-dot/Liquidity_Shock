from fastapi import FastAPI

app=FastAPI()
@app.get("/health")
def health()->dict:
    """Temporary function check karne ke liye.Delete later."""
    return {"status":"ok"}

@app.get("/ticker")
def ticker()->list[str]:
    return [
        "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS",
        "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS",
    ]

