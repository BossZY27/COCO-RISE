from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

REVISION = "0001_rise"


class Database:
    def __init__(self, url: str):
        self.engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def transaction(self):
        with self.sessions.begin() as session:
            yield session

    def ready(self) -> bool:
        with self.engine.connect() as connection:
            return connection.execute(text("SELECT version_num FROM alembic_version")).scalar() == REVISION

    def close(self):
        self.engine.dispose()
