
echo using docker image pytorch-cuda11.3: nvcr.io/nvidia/pytorch:21.04-py3

git config --global core.editor "vim"
git config --global user.name "Zhiqi Lin"
git config --global user.email "v-zhiql@microsoft.com"

sudo git config --global core.editor "vim"
sudo git config --global user.name "Zhiqi Lin"
sudo git config --global user.email "v-zhiql@microsoft.com"
sudo chmod -R a+w /opt/conda

sudo apt-get install tmux -y
wget https://raw.githubusercontent.com/zhiqi-0/EnvDeployment/master/.tmux.conf -O ~/.tmux.conf
wget https://raw.githubusercontent.com/zhiqi-0/EnvDeployment/master/.vimrc -O ~/.vimrc

echo 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

python setup.py develop
