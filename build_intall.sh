wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
sleep 5
source ~/.bashrc
conda update conda -y
conda config --add channels intel
conda create -n intel intelpython3_core python=3
source activate intel
conda install numpy scipy matplotlib pandas h5py pytables jupyter joblib numba pytest -y
pytest unittesting_scripts.py