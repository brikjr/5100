# Project Setup

This project uses Python 3.10 with PyTorch, Pandas, and NumPy. Follow the instructions below to set up your environment.

## Prerequisites

- Miniconda or Anaconda installed on your system

## Environment Setup

1. Create a new Conda environment:

```bash
conda create -n 5100 python=3.10
```

2. Activate the environment:

```bash
conda activate 5100
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Package Versions

The `requirements.txt` file specifies the following package versions:

- PyTorch: 1.10.0
- Pandas: 1.3.4
- NumPy: 1.21.4

These versions are compatible with each other and Python 3.10. If you need to update any packages, make sure to test for compatibility.

## Miniconda Installation (if needed)

If you don't have Miniconda installed, you can use the following commands:

For Apple Silicon (M1/M2) Macs:
```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

For Intel Macs or to install an older version:
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-py39_24.5.0-0-MacOSX-x86_64.sh -o ~/miniconda/miniconda.sh
bash ~/miniconda/miniconda.sh
```

After installation, initialize Conda for your shell:
```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Remember to restart your terminal after initialization.
