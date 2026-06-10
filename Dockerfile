FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV, PaddlePaddle, and Fonts
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    fonts-dejavu \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*


# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir arabic-reshaper

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Set the command to run the API
CMD ["python", "poc_api.py"]
