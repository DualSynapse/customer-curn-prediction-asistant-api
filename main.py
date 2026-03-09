import uvicorn
from src.api.main import app  # expose app agar fastapi dev bisa menemukannya

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )