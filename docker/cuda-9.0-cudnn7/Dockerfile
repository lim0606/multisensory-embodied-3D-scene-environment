FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Apply languate settings
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install some basic utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Recompile openmpi
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
RUN mkdir /src
WORKDIR /src
#RUN sudo apt-get remove openmpi-bin \
RUN wget https://download.open-mpi.org/release/open-mpi/v1.10/openmpi-1.10.3.tar.gz \
 && tar -xvzf openmpi-1.10.3.tar.gz \
 && cd openmpi-1.10.3 \
 && ./configure --prefix=/usr/local --disable-getpwuid \
 && sudo make all install

# https://github.com/openai/mujoco-py/blob/master/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libav-tools \
    libpq-dev \
    libjpeg-dev \
    cmake \
    swig \
    freeglut3 \
    libboost-all-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libsdl2-2.0-0\
    libgles2-mesa-dev \
    libsdl2-dev \
    wget \
    unzip \
    xserver-xorg-input-void \
    xserver-xorg-video-dummy \
    xpra \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# https://github.com/eric-heiden/deep-rl-docker/blob/master/Dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    nano \
    vim \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    git \
    cmake \
    zlib1g-dev \
    libjpeg-dev \
    xvfb \
    libav-tools \
    xorg-dev \
    libboost-all-dev \
    libsdl2-dev \
    swig \
    libgtk2.0-dev \
    wget \
    ca-certificates \
    unzip \
    aptitude \
    pkg-config \
    qtbase5-dev \
    libqt5opengl5-dev \
    libassimp-dev \
    libtinyxml-dev \
    golang \
    terminator \
    tmux \
    libcanberra-gtk-module \
    libfuse2 \
    libnss3 \
    fuse \
    libglfw3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    xpra \
    xserver-xorg-dev \
    libffi-dev \
    libxslt1.1 \
    libglew-dev \
    parallel \
    htop \
    apt-transport-https \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

######## install xvfb ########
# https://github.com/pascalwhoop/tf_openailab_gpu_docker/blob/master/Dockerfile
# 5. installing X and xvfb so we can SEE the action using a remote desktop access (VNC)
# and because this is the last apt, let's clean up after ourselves
RUN apt-get update && apt-get install -y --no-install-recommends \
    x11vnc \
    xvfb \
    fluxbox \
    wmctrl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
#RUN chmod -R 777 /home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.4.10-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN conda update -n base conda

######## install mujoco-py ########
# set up mujoco pro env
WORKDIR /app
RUN mkdir -p /home/user/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/user/.mujoco \
    && rm mujoco.zip
COPY ./mjkey.txt /home/user/.mujoco/
ENV LD_LIBRARY_PATH /home/user/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
#ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# install mujoco-py
WORKDIR /app
RUN git clone https://github.com/openai/mujoco-py && \
    cd mujoco-py && \
    pip install -e .
RUN chmod -R 777 .

######## install gym ########
WORKDIR /app
#RUN git clone https://github.com/openai/gym && \
RUN git clone https://github.com/lim0606/gym.git
RUN cd gym && \
    pip install -e .[all]

######## install pytorch ########
# CUDA 9.0-specific steps
RUN conda install -y -c anaconda \
    cudatoolkit=9.0 \
 && conda clean -ya
RUN conda install -y -c pytorch \
    cuda90=1.0 \
    magma-cuda90=2.3.0 \
    "pytorch=0.4.1=py36_cuda9.0.176_cudnn7.1.2_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install HDF5 Python bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install Torchnet, a high-level framework for PyTorch
RUN pip install torchnet==0.0.4

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install Graphviz
RUN pip install graphviz

# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

# Install matplotlib and ipdb
RUN pip install matplotlib ipdb gin-config seaborn>=0.9.0 sklearn
RUN pip install tensorflow-gpu tensorboardX

# Add permissions
RUN chmod -R 777 /app

# Set up mujoco-py
RUN python3 -c "import mujoco_py"
RUN python3 -c "from mujoco_py import MujocoException"


# Set the default command to python3
#CMD ["python3"]
#ENTRYPOINT ["/bin/bash"]
COPY run.sh /
#CMD ["/run.sh"]
CMD ["/run.sh", "--allow-root"]
