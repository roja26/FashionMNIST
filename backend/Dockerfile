# Use lightweight Python image
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code & model
COPY backend backend
COPY best_model.pth /app/best_model.pth
COPY data /app/data

# Expose API port
EXPOSE 5000

# Run Flask API
CMD ["python3", "backend/api.py"]
