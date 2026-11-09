"""HTTP API application."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from customer_support_rag import __version__

APPLICATION_NAME = "customer-support-rag"


class HealthResponse(BaseModel):
    status: Literal["ok"]
    application: str
    version: str


app = FastAPI(title=APPLICATION_NAME)


@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        application=APPLICATION_NAME,
        version=__version__,
    )
