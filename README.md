## Instructions for running vi-exercise code

### 1. Setup environment

Environment is setup using [Astral uv](https://docs.astral.sh/uv/). 
Installation instructions can be found [here](https://docs.astral.sh/uv/getting-started/installation/)

After uv is successully installed create the env at `.venv` with:
`>> uv sync`

### 2. Run the code

`.venv/bin/python main.py --data-folder path/to/data/folder --output-folder path/to/output/folder`

### 3. Results
Results will be figures and reports in the `train` and `test` subfolders in the output-folder, including the top_n.csv with the prioritization scores 
