FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

RUN uv run python -m spacy download en_core_web_lg

COPY app/ ./app/
COPY data/ ./data/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]