This project aims to analyze an image of a bookshelf, extract the titles, accept a reading mood from a user, and output the books that match that mood.
A video of the final deployed project can be seen here: https://www.youtube.com/watch?v=sYn1Vx80xRk&ab_channel=WinstonHeinrichs

# Project Setup

This project uses Python 3.10 with PyTorch, Pandas, Ollama, and NumPy. Follow the instructions below to set up your environment and contribute to the project.

## Run Instructions

To run this project, please run the streamlit application in src called app.py by
```bash
cd src
streamlit run app.py
```

This project requires Ollama and access to the model llama3.2-vision which can be found here: https://ollama.com/library/llama3.2-vision

Additionally, to run the app the recommendation model needs to be trained. Train this model by running the python file under src/model.py

## Prerequisites

- Miniconda or Anaconda installed on your system
- Git installed on your system
- GitHub account

## Conda Installation (if needed)

If you don't have Conda installed, you can use the following commands:

For Apple Silicon M series Macs:

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

###### Remember to restart your terminal after initialization.

### If you wanna use Discovery:
### Discovery RC login:

```bash
ssh neu_username@login.discovery.neu.edu
```
```canvas password```

- Read the discovery documentation: https://rc-docs.northeastern.edu/en/latest/
- Learn about slurm jobs from docs
- Utilize sbatch to submit jobs


## Environment Setup

1. Create a new Conda environment:

```bash
conda create -n 5100 python=3.8
```

2. Activate the environment:

```bash
conda activate 5100
```

## Package Versions

### CUDA setup

3. Install the required packages:

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1
```

```bash
pip install pandas  
```

```bash
pip install tqdm 
```

```bash
pip install tansformers 
```

```bash
pip install 'accelerate>=0.26.0'
```

## GitHub Workflow

To contribute to this project, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Create a new branch for your work:
   ```
   git checkout -b your-branch-name
   ```
   Choose a descriptive name for your branch that reflects the changes you're making.

3. Verify that you're on the correct branch:
   ```
   git branch
   ```
   This command will list all local branches, with an asterisk (*) next to the current branch.

4. Make your changes and commit them:
   ```
   git add .
   git commit -m "Descriptive commit message"
   ```

5. Push your branch to GitHub:
   ```
   git push origin your-branch-name
   ```

6. Go to the GitHub repository page and click on "Pull requests".

7. Click "New pull request".

8. Select your branch from the dropdown menu and click "Create pull request".

9. Add a title and description for your pull request, explaining the changes you've made.

10. Click "Create pull request" to submit it for review.

11. Wait for the project maintainers to review your changes. They may request modifications or approve and merge your changes into the main branch.

Remember to always pull the latest changes from the main branch before starting new work:

```
git checkout main
git pull origin main
git checkout -b your-new-branch-name
```
## Branch Protection Rules and Merging into Main

This project requires at least one review before merging changes into the main branch. This helps maintain code quality and encourages collaboration. Here's how it works:

1. When you create a pull request, it cannot be merged until at least one other contributor reviews and approves the changes.
2. The reviewer(s) will examine your code, may comment on it, and may request changes.
3. Once your pull request has been approved, you can merge it into the main branch.
