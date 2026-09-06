FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --upgrade pip \
    && pip install .

COPY notebooks/ ./notebooks/
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/processed/base_liquidez_final.csv ./data/processed/base_liquidez_final.csv
COPY README.md ./

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]