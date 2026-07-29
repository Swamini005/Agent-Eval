FROM python:3.12-slim

# Set environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy dependencies definitions
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
RUN poetry install --no-root --no-directory

# Copy the rest of the application code
COPY . .

# Run poetry install again to install the current project root package
RUN poetry install --only-root

# Expose FastAPI service port
EXPOSE 8000

# Default command to run the API service
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
