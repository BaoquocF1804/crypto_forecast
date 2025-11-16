from fastapi import FastAPI

app = FastAPI(title="Crypto Forecast API")

@app.get("/")
def health():
    return {"status": "ok"}
