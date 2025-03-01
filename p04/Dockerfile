# Use the latest Ubuntu image
FROM ubuntu:latest

# Update the package list and install dependencies
RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    cmake \
    python3 \
    htop \
    python3-pip \
    python3-numpy \
    python3-matplotlib

# Define the default command (optional)
CMD [ "bash" ]
