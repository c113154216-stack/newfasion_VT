import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    BASE_DIR = BASE_DIR
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:@127.0.0.1:3306/3d_tryon",
    )
    FAISS_INDEX_PATH = os.getenv(
        "FAISS_INDEX_PATH",
        str(BASE_DIR / "data" / "faiss_index" / "product.index"),
    )
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str(BASE_DIR / "data" / "uploads"))
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", str(BASE_DIR / "data" / "outputs"))
    FAISS_VECTOR_DIM = int(os.getenv("FAISS_VECTOR_DIM", "512"))
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
        if origin.strip()
    ]
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
