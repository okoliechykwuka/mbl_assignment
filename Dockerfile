# Train stage
FROM python:3.12-slim AS train

WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Train the model
RUN python trainer.py

# Inference stage
FROM python:3.12-slim AS inference
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trained model from the previous stage
COPY --from=train /app/model ./model

# Copy the source code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]