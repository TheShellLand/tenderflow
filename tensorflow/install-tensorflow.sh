#!/bin/bash


if [ "$1" == "--remove" ]; then
    cd /tmp
    apt purge -y cuda*
    apt purge -y libcudnn*
    apt purge -y libcuda*
    pip3 uninstall -y tensorflow
    pip3 uninstall -y tensorflow-gpu
    pip3 uninstall -y tensorboard
    pip3 uninstall -y keras
    pip3 uninstall -y h5py
    rm -r /usr/local/cuda-8.0
    exit 0
fi

cd $(dirname "$0")

# Install Tensorflow for python3 + keras
venv=/tmp/tensorflow
mkdir $venv
apt install -y python3-pip python3-dev python-virtualenv
virtualenv -p python3 $venv
# virtualenv --system-site-packages -p python3 tensorflow
pip3 install --upgrade pip
# Activate venv
# source $venv/bin/activate
# Tensorflow CPU-only
pip3 install --upgrade tensorflow
pip3 install --upgrade tensorboard
# Tensorflow + GPU
if [ "$1" == "--gpu" ]; then
    # http://us.download.nvidia.com/XFree86/Linux-x86_64/384.59/NVIDIA-Linux-x86_64-384.59.run
    # https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
    # https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64-deb
    # https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7/prod/8.0_20170802/cudnn-8.0-linux-x64-v7-tgz
    # https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7/prod/8.0_20170802/Ubuntu16_04_x64/libcudnn7_7.0.1.13-1+cuda8.0_amd64-deb
    # https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7/prod/8.0_20170802/Ubuntu16_04_x64/libcudnn7-dev_7.0.1.13-1+cuda8.0_amd64-deb
    add-apt-repository -y ppa:graphics-drivers/ppa
    apt update
    apt install -y --reinstall nvidia-375

    dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    dpkg -i cuda-repo-ubuntu1604-8-0-local-cublas-performance-update_8.0.61-1_amd64.deb

    # cuDNN 6
    dpkg -i libcudnn6_6.0.21-1+cuda8.0_amd64.deb
    dpkg -i libcudnn6-dev_6.0.21-1+cuda8.0_amd64.deb
    cp -vP cuda/include/cudnn.h /usr/local/cuda/include/
    cp -vP cuda/lib64/libcudnn*  /usr/local/cuda/lib64/
#    tar zxf processed-cudnn-8.0-linux-x64-v6.tgz -C /

    # cuDNN 7
#    dpkg -i libcudnn7_7.0.1.13-1+cuda8.0_amd64.deb
#    dpkg -i libcudnn7-dev_7.0.1.13-1+cuda8.0_amd64.deb
#    tar zxf processed-cudnn-8.0-linux-x64-v7.tgz -C /

    apt update
    apt install -y --reinstall cuda
    apt install -y --reinstall libcupti-dev
    pip3 install --upgrade tensorflow-gpu
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/
fi
# If the above failed
# tfBinaryURL  =>  URLs
# https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package
#pip3 install --upgrade tfBinaryURL
# Keras
pip3 install --upgrade keras
# h5py
pip3 install --upgrade h5py
#deactivate

echo "Done"

echo "Run: source $venv/bin/activate"

if [ -z "$1" ]; then
    echo "For GPU support: install-tensorflow.sh --gpu"
    echo "Remove: install-tensorflow.sh --remove"
fi
