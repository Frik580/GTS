"""Shared aiohttp SSL/connector helpers for external API calls."""
from __future__ import annotations

import logging
import ssl

import aiohttp

import config

logger = logging.getLogger("GTS.HTTP")


def create_verified_ssl_context() -> ssl.SSLContext:
    """Build a verified context using both OS roots and certifi when available."""
    ctx = ssl.create_default_context()
    try:
        import certifi
        ctx.load_verify_locations(cafile=certifi.where())
    except ImportError:
        pass
    return ctx


def create_ssl_context() -> ssl.SSLContext:
    if not config.SSL_VERIFY:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return create_verified_ssl_context()


def create_connector() -> aiohttp.TCPConnector:
    return aiohttp.TCPConnector(ssl=create_ssl_context())


def create_session(**kwargs) -> aiohttp.ClientSession:
    if "connector" not in kwargs:
        kwargs["connector"] = create_connector()
    return aiohttp.ClientSession(**kwargs)
