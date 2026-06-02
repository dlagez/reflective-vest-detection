"""FastAPI 应用工厂."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Reflective Vest Detection API",
        description="API for reflective vest compliance detection",
        version="1.0.0",
    )

    from src.api.routes import router
    app.include_router(router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()
