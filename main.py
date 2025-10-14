# main.py
from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from config import Settings, configure_logging, service_meta, tz
from runtime import Runtime

settings = Settings()
log = configure_logging().bind(module="main")
app = FastAPI()

# install metrics
try:
    Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")
except Exception:
    pass


@app.on_event("startup")
async def _startup():
    rt = Runtime(settings=settings, tz=tz(settings), log=log)
    await rt.start()
    app.state.runtime = rt


@app.on_event("shutdown")
async def _shutdown():
    rt: Runtime | None = getattr(app.state, "runtime", None)
    if rt:
        await rt.stop()


@app.get("/sigma")
async def sigma():
    rt: Runtime = app.state.runtime
    return await rt.sigma_payload()


@app.get("/healthz")
async def healthz():
    rt: Runtime = app.state.runtime
    return await rt.health_payload()


@app.get("/version")
async def version():
    return service_meta()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
