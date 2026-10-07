"""Central paths and local service settings."""

import os
from pathlib import Path

from dotenv import load_dotenv


def project_root():
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Run from the project directory or set PROJECT_ROOT")


ROOT = project_root()
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def database_url():
    return os.getenv("DATABASE_URL", "postgresql+psycopg://retail:retail@localhost:5433/retail")
