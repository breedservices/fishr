from sys import platform

if platform != "win32":
    try:
        from uvloop import install

        install()
    except ImportError:
        pass

import asyncio

__all__ = ["asyncio"]
