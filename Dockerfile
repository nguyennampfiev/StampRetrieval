# Use the official Python base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install necessary build tools
RUN apt-get update && \
    apt-get install -y git g++ make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install specific versions of PyTorch and torchvision
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 torchaudio --extra-index-url https://download.pytorch.org/whl/cpu numpy==1.24.4

RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.1.0"
RUN git clone https://github.com/open-mmlab/mmdetection.git

# Set the working directory to the cloned mmdetection directory
WORKDIR /app/mmdetection

# Install MMDetection from source
RUN pip install -v -e .
# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK and download required data
RUN pip install nltk && \
    python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
# Copy the rest of your app code into the container
COPY . .

# Set the default command to run your app
CMD ["streamlit", "run", "app.py"]
