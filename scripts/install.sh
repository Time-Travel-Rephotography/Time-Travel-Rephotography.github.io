# conda create -n stylegan python=3.7
# conda activate stylegan
conda install -c conda-forge/label/gcc7 opencv --yes
conda install tensorflow-gpu=1.15 cudatoolkit=10.0 --yes
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch --yes
pip install -r requirements.txt
