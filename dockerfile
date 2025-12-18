# Use a lightweight official Python image
FROM python:3.10-slim

# Set the working directory inside the container to /app
# All commands will run relative to this directory
WORKDIR /app

# Copy the dependency list first
# This allows Docker to cache installed packages
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir keeps the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY app/ .

# Command that runs when the container starts
# Executes the main real-time stock prediction script
CMD ["python", "main.py"]
