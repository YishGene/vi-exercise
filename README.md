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

*
*
*

# Documentation

The flow of the code is: 
1. Data ingestion
2. Featurization
3. Training CATE model
4. Evaluation and output

Some explanations: 
### 1. Data ingestion
Each csv has its own routine with various filtrations. 

Web visits - I dropped the description and the url, there are 26 unique sites and I chose events belonging only to the relevant domains from this list of sites. 

All dates were converted to timestamps, then filtered by the observation window (this is a safety precation for new data sources - all the data were within the observation window). Null values were dropped. 

### 2. Featurization
Features are an aggregation per member_id of different events. Aggregations are:
-  time since first event
-  time since last event
-  event count

In the Web Visits csv, I filtered out web visits to unrelated topics. 

In the Claims csv, I tried: 
- aggregating on single claims codes
- aggregating all claims together
- filtered on the claims that are on the client brief (E11.9, I10, and Z71.3). 
 
In the end the best result was on the aggregated all claims together. 

Merging together, some member_ids had no inputs for some of the data - theses were filled according to relevant information (e.g., counts filled with 0)

### 3. CATE model

Since this is a treatement effect, and we cannot observe the counterfactual for each training member_id, CATE is the model chosen for this problem. 

We are modeling the difference between the expected effect with and without treatment using an XLearner. This models `E[Y|T=1, X] - E[Y|T=0, X]` which gives us for each member the expected difference if a treatment (outreach) is given. The underlying model is LightGBM, with some degree of regularization, both tree depth, L1, L2, and number of trees. 

This CATE approach naturally incorporates the outreach parameter into the model, each underlying LighGBM model models the effect with and without outreach. 

### 4. Evaluation and output
Because of the choice of CATE as the method, I looked at the appropriate Qini graph and the AUUC (Area under the uplift curve) to measure the quality of the outreach effect prediction. 

Outreach size (n) is taken frorm the first elbow of the Qini curve, estimated at ~1,600 members out of 10,000. This is the point in the graph where the additional increment in outcome starts to flatten out relative to the number of outreaches acheived. 

Unfortunately, due to severe overfitting which I could not get rid of, this can only be seen in the training set and not the test set, so it probably will have no real effect. 
