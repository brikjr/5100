# Project Setup

This project uses Python 3.10 with PyTorch, Pandas, and NumPy. Follow the instructions below to set up your environment and contribute to the project.

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

## Environment Setup

1. Create a new Conda environment:

```bash
conda create -n 5100 python=3.10
```

2. Activate the environment:

```bash
conda activate 5100
```

## Package Versions

### THERE IS NO `requirements.txt` YET SKIP THIS STEP
The `requirements.txt` file specifies the following package versions

3. Install the required packages:

```bash
pip install -r requirements.txt
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
