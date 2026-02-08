# Development container for Semantic SLAM thesis
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential cmake git wget curl \
    # ORB-SLAM3 dependencies
    libglew-dev libgl1-mesa-dev libglu1-mesa-dev \
    libeigen3-dev libopencv-dev libssl-dev \
    libboost-all-dev libepoxy-dev \
    # Pangolin dependencies
    libglfw3-dev libxkbcommon-dev libwayland-dev \
    # Python
    python3.10 python3.10-dev python3-pip \
    # Utilities
    vim nano htop tmux \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


# Create workspace
WORKDIR /workspace
# Set up environment
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Default command
CMD ["/bin/bash"]