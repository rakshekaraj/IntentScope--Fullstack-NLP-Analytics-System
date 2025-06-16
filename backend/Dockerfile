# backend/Dockerfile

FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy backend code and model files
COPY ./backend /app
COPY ./model /app/model

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
