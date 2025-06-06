# Setting Up the Development Environment

Follow these steps to set up the development environment:

1. **Install Anaconda**

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
5. **Run the llama server**
   Install llama.cpp and save model to models/base/
   cd into folder root:
   ```bash
   llama-server -m /Users/yash/Code/Personal/IBM/models/base/Llama-3.2-3B-Instruct-F16.gguf
   ```
6. **Populate chromadb**
   ```bash
   python src/rag/vectorstore/load_data.py
   ```
7. **Launch RAG**
   ```bash
   python src/rag/app.py
   ```