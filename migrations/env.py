from alembic import context
from sqlalchemy import create_engine
from app.core.settings import Settings
from app.infrastructure.postgres.models import Base


def run():
    url = Settings().database_url
    if not url:
        raise RuntimeError("Set RISE_DATABASE_URL before running migrations")
    if context.is_offline_mode():
        context.configure(url=url, target_metadata=Base.metadata, literal_binds=True, include_schemas=True)
        with context.begin_transaction():
            context.run_migrations()
    else:
        engine = create_engine(url)
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=Base.metadata, include_schemas=True)
            with context.begin_transaction():
                context.run_migrations()
        engine.dispose()

run()
