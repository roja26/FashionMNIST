# Use lightweight Python image
FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY frontend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend code
COPY frontend frontend

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
