FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    # sympy/mpmath are pulled in by onnxruntime for optional symbolic shape
    # inference; plain InferenceSession.run() doesn't touch that path --
    # ~85MB not worth shipping. Stripped here (builder stage) rather than
    # after the COPY below, since deleting post-copy only masks the layer,
    # it doesn't shrink the final image.
    rm -rf /root/.local/lib/python3.11/site-packages/sympy \
           /root/.local/lib/python3.11/site-packages/mpmath

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
