FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Generate synthetic dataset if missing (required for FL client pods)
RUN if [ ! -f dataset/clients/client_1/clinical.csv ] || [ ! -f dataset/clients/client_1/us_embeddings.npy ]; then python data/synthetic_generator.py; fi

# Expose ports for Flower server and Streamlit dashboard
EXPOSE 8080 8501

# Command to run (defaults to Streamlit dashboard)
# To run Flower server: CMD ["python", "federated/server.py", "5"]
# To run Flower client: CMD ["python", "federated/client.py", "1"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
