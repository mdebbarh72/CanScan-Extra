
FROM apache/airflow:3.3.2-python3.11

COPY airflow/requirements.txt /requirements.txt

RUN pip install --no-cache-dir \
    "apache-airflow==3.3.2" \
    -r /requirements.txt

# Install CPU-only PyTorch to avoid NVIDIA CUDA dependencies.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install Ultralytics after CPU-only PyTorch.
RUN pip install --no-cache-dir ultralytics
