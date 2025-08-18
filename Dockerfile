# PhenoTagger FastAPI Dockerfile
FROM tensorflow/tensorflow:2.12.0-gpu

# Install basic tools + Java (OpenJDK 11)
RUN apt-get update && apt-get install -y \
    git unzip openjdk-11-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Download all popular NLTK resources (including punkt, wordnet, pos_tag, etc.)
RUN python -c "import nltk; nltk.download('popular', quiet=False)"

# Copy application code
COPY . /PhenoTagger
WORKDIR /PhenoTagger/src/

# Create necessary directories for models and data
RUN mkdir -p /PhenoTagger/models /data/input /data/output

# # Copy the FastAPI app and configuration files to src directory
# COPY app.py config.py phenotagger_api.py .env* ./

# Expose port for FastAPI
EXPOSE 8000

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]