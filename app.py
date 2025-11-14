# app.py
import os
import sys
import socket
import joblib
import webbrowser
import threading
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

# Try to use ngrok for public URL
import shutil
NGROK_AVAILABLE = shutil.which("ngrok") is not None
NGROK_PATH = shutil.which("ngrok") if NGROK_AVAILABLE else None

# Also try pyngrok as fallback
try:
    from pyngrok import ngrok as pyngrok_module
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False

# -------- Paths --------
VECTORIZER_PATH = os.path.join("artifacts", "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join("artifacts", "sentiment_model.pkl")
ENCODER_PATH = os.path.join("artifacts", "sentiment_encoder.pkl")

# -------- Load artifacts --------
try:
    vectorizer = load_object(VECTORIZER_PATH)
    model = load_object(MODEL_PATH)
    encoder = load_object(ENCODER_PATH)
except Exception as e:
    raise CustomException(e, sys)

# -------- FastAPI --------
app = FastAPI(
    title="Twitter Sentiment API",
    description="Simple API for sentiment prediction",
    version="1.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class TweetIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    label_index: int
    scores: dict

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

# -------- Web Routes --------
@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    with open("templates/dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())


# -------- API Routes --------
@app.post("/predict", response_model=PredictionOut)
def predict(payload: TweetIn):
    try:
        text = payload.text
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        X = vectorizer.transform([text])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            pred = model.predict(X)
            probs = np.zeros(len(encoder.classes_))
            probs[pred[0]] = 1.0

        idx = int(np.argmax(probs))
        label = encoder.inverse_transform([idx])[0]

        labels = encoder.inverse_transform(range(len(encoder.classes_)))
        score_dict = {cls: float(p) for cls, p in zip(labels, probs)}

        return PredictionOut(label=str(label), label_index=idx, scores=score_dict)

    except Exception as e:
        raise CustomException(e, sys)


# -------- Find free port --------
def find_free_port(start_port: int = 8000) -> int:
    """Find a free port starting from start_port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# -------- Start server --------
if __name__ == "__main__":
    # Find a free port
    port = find_free_port()
    url = f"http://localhost:{port}"
    docs_url = f"{url}/docs"
    
    # Create public tunnel with ngrok if available
    public_url = None
    if NGROK_AVAILABLE or PYNGROK_AVAILABLE:
        try:
            import subprocess
            import json
            import time
            
            # Kill any existing ngrok processes
            try:
                subprocess.run(["pkill", "-f", "ngrok"], capture_output=True)
                time.sleep(1)
            except:
                pass
            
            # Start ngrok using CLI
            if NGROK_AVAILABLE:
                # Start ngrok in background
                ngrok_process = subprocess.Popen(
                    [NGROK_PATH, "http", str(port), "--log=stdout"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(3)  # Wait for ngrok to start
                
                # Get public URL from ngrok API
                try:
                    import urllib.request
                    with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as response:
                        data = json.loads(response.read().decode())
                        if data.get("tunnels"):
                            public_url = data["tunnels"][0]["public_url"]
                except:
                    # Fallback to pyngrok if CLI method fails
                    if PYNGROK_AVAILABLE:
                        try:
                            ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
                            if ngrok_token:
                                pyngrok_module.set_auth_token(ngrok_token)
                            public_tunnel = pyngrok_module.connect(port, bind_tls=True)
                            public_url = public_tunnel.public_url
                        except:
                            pass
            elif PYNGROK_AVAILABLE:
                # Use pyngrok as fallback
                ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
                if ngrok_token:
                    pyngrok_module.set_auth_token(ngrok_token)
                public_tunnel = pyngrok_module.connect(port, bind_tls=True)
                public_url = public_tunnel.public_url
            
            if public_url:
                # Save public URL to file for easy access
                with open("public_url.txt", "w") as f:
                    f.write(public_url)
                print(f"\n{'='*60}")
                print(f"🌐 PUBLIC URL: {public_url}")
                print(f"   Share this link to access from anywhere!")
                print(f"   URL also saved to: public_url.txt")
                print(f"{'='*60}")
        except Exception as e:
            error_msg = str(e).lower()
            print(f"\n⚠️  Could not create public tunnel: {e}")
            if "certificate" in error_msg or "ssl" in error_msg:
                print("   💡 SSL certificate issue detected.")
                print("   💡 Try: pip install --upgrade certifi")
                print("   💡 Or install ngrok manually: brew install ngrok/ngrok/ngrok")
            elif "authtoken" in error_msg or "unauthorized" in error_msg:
                print("   💡 To enable public URLs, get a free ngrok token:")
                print("      1. Sign up at https://dashboard.ngrok.com/signup")
                print("      2. Get your authtoken from the dashboard")
                print("      3. Set it: export NGROK_AUTH_TOKEN='your_token_here'")
                print("      4. Or run: ngrok config add-authtoken YOUR_TOKEN")
            print("   Continuing with localhost only...\n")
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting FastAPI server on port {port}")
    print(f"📍 Local URL: {url}")
    if public_url:
        print(f"🌐 Public URL: {public_url}")
    else:
        print(f"💡 Tip: Install pyngrok for public URL: pip install pyngrok")
    print(f"📖 API Documentation: {docs_url}")
    print(f"📚 Alternative docs: {url}/redoc")
    print(f"{'='*60}\n")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(1.5)  # Wait for server to start
        # Open public URL if available, otherwise local
        browser_url = public_url if public_url else url
        webbrowser.open(browser_url)
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start the server
    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except KeyboardInterrupt:
        if NGROK_AVAILABLE and public_url:
            try:
                ngrok.kill()
                print("\n✅ Ngrok tunnel closed")
            except:
                pass
