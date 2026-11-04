from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migrations_have_one_deployable_head() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))
    assert scripts.get_current_head() is not None
