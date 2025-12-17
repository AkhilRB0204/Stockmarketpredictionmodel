# Start from a lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the dependencies first (helps with caching)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the app code into the container
COPY app/ app/

# Default command: run the real-time predictor
CMD ["python", "app/main.py"]
