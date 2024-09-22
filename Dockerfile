# Use the official Python image from the Docker Hub
FROM python:3.10

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    || { echo "Package installation failed. Retrying in 3 seconds..."; sleep 3; apt-get update && apt-get install -y poppler-utils; }

COPY . /app

WORKDIR /app

# Install the requirements from the requirements.txt
RUN pip install -r requirements.txt


# Make port 7860 available to the world outside this container
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py"]

