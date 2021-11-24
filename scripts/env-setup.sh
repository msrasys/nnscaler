
echo using docker image pytorch-cuda11.3: pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

git config --global core.editor "vim"
git config --global user.name "Zhiqi Lin"
git config --global user.email "v-zhiql@microsoft.com"

sudo git config --global core.editor "vim"
sudo git config --global user.name "Zhiqi Lin"
sudo git config --global user.email "v-zhiql@microsoft.com"
sudo chmod -R a+w /opt/conda

sudo apt-get install tmux -y
sudo apt-get install psmisc -y
sudo apt-get install lsof -y

# install blob
# sudo apt-get install lsb-release -y
# wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
# sudo dpkg -i packages-microsoft-prod.deb
# sudo apt-get update
# sudo apt-get install blobfuse -y
# sudo rm packages-microsoft-prod.deb

# install azcopy
wget https://azcopyvnext.azureedge.net/release20210616/azcopy_linux_amd64_10.11.0.tar.gz -O azcopy.tar.gz
tar -zxvf azcopy.tar.gz
sudo mv azcopy_linux_amd64_10.11.0/azcopy /usr/bin/
rm -rf azcopy_linux_amd64_10.11.0 azcopy.tar.gz

wget https://raw.githubusercontent.com/zhiqi-0/EnvDeployment/master/.tmux.conf -O ~/.tmux.conf
wget https://raw.githubusercontent.com/zhiqi-0/EnvDeployment/master/.vimrc -O ~/.vimrc

echo 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc

# cmd for count code lines
# find cube/ -name "*.py" -print0 | xargs -0 wc -l
pip uninstall training_daemon
python setup.py develop
pip install -r requirements.txt

