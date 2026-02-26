import os

from celery import Celery


def create_celery() -> Celery:
    celery_app = Celery(
        "scam_defender_tasks",
        broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
    )

    return celery_app


celery = create_celery()


@celery.task(name="scam_defender.cleanup_old_scans")
def cleanup_old_scans(retention_days: int = 90) -> int:
    from app import create_app

    app = create_app()
    with app.app_context():
        store = app.extensions["store"]
        summary = store.cleanup_old_records(retention_days=max(7, int(retention_days)))
        return int(summary.get("scan_records_deleted", 0))


@celery.task(name="scam_defender.model_health_snapshot")
def model_health_snapshot() -> dict:
    from app import create_app

    app = create_app()
    with app.app_context():
        service = app.extensions["model_service"]
        store = app.extensions["store"]
        return {
            "models": service.get_status(),
            "mongo_connected": store.ping(),
        }
