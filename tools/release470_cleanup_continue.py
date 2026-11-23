"""Provider-free cleanup of only the owned continuation FAQ after a killed worker."""

import asyncio
import json
import logging
import os
from pathlib import Path


async def cleanup():
    import importlib.util

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name(filename)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cleanup module missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    continuation = load("eval470_continuation", "release470_continue.py")
    helper = load("eval470_faq_cleanup", "release470_faq_api.py")
    return await helper.operate(
        "cleanup",
        manifest_path=Path("/reports/faq-continue-records-20261123.json"),
        application=continuation.faq_application(),
    )


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    try:
        print(json.dumps(asyncio.run(cleanup()), indent=2))
    except BaseException as error:
        print(json.dumps({"failed": type(error).__name__}))
        raise SystemExit(1) from None
