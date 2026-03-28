FROM python:3.12-slim

COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir ".[dev]"

ENTRYPOINT ["python3", "-m", "pytest", "tests/integration/", "-v", "--tb=short"]
