# Use python 3.11 lightweight version
FROM python:3.11-slim

# Set folder inside the container
WORKDIR /app

# Install system tools needed for building libraries
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your actual code
COPY . .

# Create folders for logs and database
RUN mkdir -p logs storage/faiss_index

# Allow traffic on port 8000
EXPOSE 8000

# The command to start the backend server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]