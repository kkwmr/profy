# Specify the base image with CUDA
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Proceed with the installation of Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip libsndfile1

# Set the working directory inside the container
WORKDIR /workspace

# Add the 'src' directory to the Python module search path
ENV PYTHONPATH=/workspace/ddsupport

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir jupyter

# Expose the port the app runs on
EXPOSE 8888

# Use a shell as the default command
CMD ["/bin/bash"]