# Setting Up the Development Environment

Follow these steps to set up the development environment:

1. **Install Anaconda or Miniconda**
   If you don't already have Anaconda or Miniconda installed, download and install it from [here](https://www.anaconda.com/products/distribution).

2. **Create a Conda Environment**
   Run the following command to create a new Conda environment named `dev` with Python 3.12.6:
   ```bash
   conda create -n dev python=3.12.6 -y
   ```

3. **Activate the Environment**
   Activate the environment using:
   ```bash
   conda activate dev
   ```

4. **Install Dependencies**
   Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

You are now ready to start working on the project!