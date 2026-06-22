from msgspec import Struct
from primp import Client as PrimpClient


class Request(Struct, frozen=True):
    url: str
    query: dict | None = None
    headers: dict | None = None
    cookies: dict | None = None
    timeout: float | None = None
    impersonate: str | None = None


class BodyRequest(Struct, frozen=True):
    url: str
    json: dict | None = None
    data: dict | None = None
    query: dict | None = None
    headers: dict | None = None
    cookies: dict | None = None
    timeout: float | None = None
    impersonate: str | None = None


def _pick(req: Request | BodyRequest) -> str:
    if req.impersonate is not None:
        return req.impersonate
    return "random"


def make_client(
    impersonate: str = "chrome", cookie_store: bool = True, headers: dict | None = None
) -> PrimpClient:
    return PrimpClient(
        impersonate=impersonate,
        cookie_store=cookie_store,
        headers=headers,
    )


def get(req: Request) -> object:
    return PrimpClient(impersonate=_pick(req)).get(
        req.url,
        params=req.query,
        headers=req.headers,
        cookies=req.cookies,
        timeout=req.timeout,
    )


def post(req: BodyRequest) -> object:
    return PrimpClient(impersonate=_pick(req)).post(
        req.url,
        json=req.json,
        data=req.data,
        params=req.query,
        headers=req.headers,
        cookies=req.cookies,
        timeout=req.timeout,
    )


def put(req: BodyRequest) -> object:
    return PrimpClient(impersonate=_pick(req)).put(
        req.url,
        json=req.json,
        data=req.data,
        params=req.query,
        headers=req.headers,
        cookies=req.cookies,
        timeout=req.timeout,
    )


def delete(req: Request) -> object:
    return PrimpClient(impersonate=_pick(req)).delete(
        req.url,
        params=req.query,
        headers=req.headers,
        cookies=req.cookies,
        timeout=req.timeout,
    )


def patch(req: BodyRequest) -> object:
    return PrimpClient(impersonate=_pick(req)).patch(
        req.url,
        json=req.json,
        data=req.data,
        params=req.query,
        headers=req.headers,
        cookies=req.cookies,
        timeout=req.timeout,
    )


def head(req: Request) -> object:
    return PrimpClient(impersonate=_pick(req)).head(
        req.url,
        params=req.query,
        headers=req.headers,
        cookies=req.cookies,
        timeout=req.timeout,
    )


__all__ = [
    "get",
    "post",
    "put",
    "delete",
    "patch",
    "head",
    "make_client",
    "Request",
    "BodyRequest",
]
