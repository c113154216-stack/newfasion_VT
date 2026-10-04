import io
import os

import pytest
from PIL import Image

# Pre-set env vars before any import of app/config so the module-level
# create_app() in app.py doesn't attempt to open a real FAISS index file.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("FAISS_INDEX_PATH", "")

from app import create_app  # noqa: E402
from database.models import Base  # noqa: E402


@pytest.fixture
def faiss_path(tmp_path):
    return str(tmp_path / "test.index")


@pytest.fixture
def app(faiss_path, tmp_path, monkeypatch):
    """Create an isolated Flask test app with SQLite in-memory DB and temp FAISS index."""
    upload_folder = str(tmp_path / "uploads")
    output_folder = str(tmp_path / "outputs")

    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("FAISS_INDEX_PATH", faiss_path)
    monkeypatch.setenv("UPLOAD_FOLDER", upload_folder)
    monkeypatch.setenv("OUTPUT_FOLDER", output_folder)

    import config
    import services.vector_store as vs

    monkeypatch.setattr(config.Config, "DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setattr(config.Config, "FAISS_INDEX_PATH", faiss_path)
    monkeypatch.setattr(config.Config, "UPLOAD_FOLDER", upload_folder)
    monkeypatch.setattr(config.Config, "OUTPUT_FOLDER", output_folder)

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    test_engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)

    import database.models as models

    monkeypatch.setattr(models, "engine", test_engine)
    monkeypatch.setattr(models, "SessionLocal", TestSession)

    # Reset singleton so each test gets a fresh VectorStore
    vs._vector_store = None

    application = create_app()
    application.config["TESTING"] = True
    yield application

    # Cleanup: reset singleton after test
    vs._vector_store = None


@pytest.fixture
def client(app):
    return app.test_client()


def make_test_image():
    """Create a minimal in-memory JPEG for upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(255, 255, 255)).save(buf, format="JPEG")
    buf.seek(0)
    return buf
