FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system agri && useradd --system --gid agri --create-home agri

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts

RUN python -m pip install --upgrade pip && python -m pip install .

RUN mkdir -p /data && chown -R agri:agri /app /data

USER agri

ENTRYPOINT ["python", "-m", "agri_auditor"]
CMD ["process", "--data-dir", "/data", "--output-features", "/data/features.csv", "--output-events", "/data/events.json", "--output-report", "/data/audit_report.html"]
