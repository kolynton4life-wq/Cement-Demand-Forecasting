"""
DB connection helper.
Even though SQLite needs no server/credentials, we route through
SQLAlchemy so swapping to Postgres/SQL Server later (e.g. if MIG
moves this off a flat file) only touches this one function.
"""
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DB_PATH


def get_engine() -> Engine:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            f"Copy MIG_Cement_Records.db into data/raw/ first."
        )
    return create_engine(f"sqlite:///{DB_PATH}")


if __name__ == "__main__":
    engine = get_engine()
    with engine.connect() as conn:
        from sqlalchemy import text
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table';"))
        print("Connected OK. Tables:", [r[0] for r in result])
