import os
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse
from starlette.status import HTTP_401_UNAUTHORIZED
from core.scalper_ai import ScalperAI

API_TOKEN = os.environ.get("BOT_API_TOKEN")
if not API_TOKEN:
    raise RuntimeError("BOT_API_TOKEN environment variable not set.")

app = FastAPI(title="Legendary Ultra ScalperAI Remote API")

scalper = ScalperAI()

# --- Auth Dependency ---
def verify_token(request: Request):
    token = request.headers.get("Authorization")
    if not token or token != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Unauthorized")

@app.get("/health", dependencies=[Depends(verify_token)])
def health():
    return {"status": "ok"}

@app.get("/status", dependencies=[Depends(verify_token)])
def status():
    return {
        "daily_loss": scalper.daily_loss,
        "consecutive_losses": scalper.consecutive_losses,
        "last_trade_day": scalper.last_trade_day,
        "stake_strategy": scalper.stake_strategy,
        "current_asset": scalper.current_asset,
        "api_fail_count": scalper.api_fail_count,
        "heartbeat": scalper.last_heartbeat,
        "crash_recovery_attempts": scalper.crash_recovery_attempts,
    }

@app.get("/download/log", dependencies=[Depends(verify_token)])
def download_log():
    log_path = "trade_log.csv"
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Log file not found.")
    return FileResponse(log_path, filename="trade_log.csv", media_type="application/octet-stream")

# Add more endpoints as needed (start/stop, rotate key, set params, etc.)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("remote_api:app", host="0.0.0.0", port=8000, reload=True)
