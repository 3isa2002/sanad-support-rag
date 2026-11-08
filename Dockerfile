FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY mock_backend ./mock_backend
RUN pip install --no-cache-dir .

COPY data/kb ./data/kb
COPY data/mock_backend ./data/mock_backend
COPY data/eval ./data/eval

EXPOSE 8000
CMD ["sanad", "serve", "--host", "0.0.0.0", "--port", "8000"]
