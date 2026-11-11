FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd --uid 10001 --create-home dataclean && mkdir -p /app/data/files && chown -R dataclean:dataclean /app
COPY --chown=dataclean:dataclean app ./app
COPY --chown=dataclean:dataclean migrations ./migrations
COPY --chown=dataclean:dataclean examples ./examples
COPY --chown=dataclean:dataclean alembic.ini ./
USER dataclean
EXPOSE 8000
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
USER root
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY --chown=dataclean:dataclean tests ./tests
USER dataclean
CMD ["python", "-m", "pytest", "-q"]
