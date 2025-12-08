# Kaggle AI Agent
A simple AI agent that can search and download datasets from kaggle and then perform basic data analysis using profiler and provides an html report.

## Setup
1. Clone the repository:
   ```bash
   git clone
    ```
2. Navigate to the project directory.
3. Install the required dependencies:
   ```bash
   uv add -r requirements.txt
   ```
4. Set up Kaggle API credentials:
    - Create a `kaggle.json` file with your Kaggle API credentials.
    - Place the `kaggle.json` file in the: ~/.kaggle/ directory.

5. Api key setup for Groq:
   - Set your GROQ API key as an environment variable:
   ```bash
   export GROQ_API_KEY='your_groq_api_key'
   ``` 

## Usage
1. Run the AI agent script:
    ```bash
    uvicorn app.main:app
    ```

2. Access the AI agent via the provided endpoint (e.g., http://localhost:8000).
3. Provide the dataset search query and analysis instructions through the API.
4. The agent will search for the dataset on Kaggle, download it, perform data analysis, and generate an HTML report.
