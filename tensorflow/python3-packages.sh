
venv=~/.venv/tensorflow
mkdir -p $venv
virtualenv -p python3 $venv
source $venv/bin/activate
sudo -H  pip3 install --upgrade pip
sudo -H  pip3 install --upgrade tensorflow
sudo -H  pip3 install --upgrade tensorflow-gpu
sudo -H  pip3 install --upgrade tensorboard
sudo -H  pip3 install --upgrade keras
sudo -H  pip3 install --upgrade h5py

pip3 install --upgrade pip
pip3 install --upgrade tensorflow
pip3 install --upgrade tensorboard
pip3 install --upgrade keras
pip3 install --upgrade h5py