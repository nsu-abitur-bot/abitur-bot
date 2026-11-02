"""Explicit local first-admin provisioning: uv run python -m api.auth.bootstrap."""

import argparse
import asyncio
import getpass
import sys
import warnings

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from api.auth.security import hash_password
from db.postgres.config import get_database_url
from db.postgres.services.admin import AdminService, BootstrapRefusedError


async def provision(username: str, password: str) -> None:
    engine = create_async_engine(get_database_url(), echo=False, hide_parameters=True)
    try:
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory.begin() as session:
            await AdminService(session).bootstrap_superadmin(
                username, hash_password(password)
            )
    finally:
        await engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the first local superadmin")
    parser.add_argument("--username", required=True)
    args = parser.parse_args()
    try:
        # Never fall back to echoed stdin when a controlling terminal is unavailable.
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            password = getpass.getpass("Password: ")
            confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            print("Passwords do not match", file=sys.stderr)
            return 1
        # Do not print validation errors: they include the submitted password.
        if not 3 <= len(args.username) <= 100 or len(password) < 8:
            print(
                "Username must be 3–100 characters; password at least 8", file=sys.stderr
            )
            return 1
        asyncio.run(provision(args.username, password))
    except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
        print("Secure password input cancelled or unavailable", file=sys.stderr)
        return 1
    except BootstrapRefusedError:
        print("Bootstrap refused: admins already exist", file=sys.stderr)
        return 1
    except Exception:
        print(
            "Bootstrap failed; check database configuration and migrations",
            file=sys.stderr,
        )
        return 1
    print("First superadmin created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
