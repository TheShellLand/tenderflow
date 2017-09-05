
venv=~/.venv/tensorflow
mkdir -p $venv
virtualenv -p python3 $venv
source $venv/bin/activate
sudo -H  pip3 install --upgrade pip
sudo -H  pip3 install --upgrade tensorflow
sudo -H  pip3 install --upgrade tensorflow-gpu
sudo -H  pip3 install --upgrade keras
sudo -H  pip3 install --upgrade h5py
