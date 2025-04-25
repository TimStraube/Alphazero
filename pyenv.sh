#!/bin/bash

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Install pyenv
curl https://pyenv.run | bash

# Set up pyenv in shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

# Load pyenv without restarting shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv install 3.11.1

# Create alphazero virtual environment
pyenv virtualenv 3.11.1 alphazero

# Navigate to project directory
cd /home/aisyslab/Programme/Alphazero

# Set local Python version to the alphazero virtual environment
pyenv local alphazero

echo "Setup complete! The 'alphazero' virtual environment is now active in this directory."
echo "To install packages, use: pip install <package-name>"