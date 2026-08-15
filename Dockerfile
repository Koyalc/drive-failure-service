FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY src/ ./src/
COPY artifacts/ ./artifacts/
USER appuser
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PORT=8080
CMD exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}
