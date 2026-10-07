FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.6
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN uv sync --frozen --no-editable --no-dev
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
ENV PATH="/app/.venv/bin:$PATH" PROJECT_ROOT=/app
CMD ["uvicorn", "retail_pricing.api:app", "--host", "0.0.0.0", "--port", "8000"]
