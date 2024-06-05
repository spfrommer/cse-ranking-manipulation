# Overview
This repository reproduces the results for the paper "Ranking Manipulation for Conversational Search Engines."

For convenience, we also release the pickled output (`out`), raw-text output (`out_text`), dataset (`dataset`), and plots (`plots`). **Note that the repository size is thus rather large (~100Mb as zip files, ~700Mb after unzipping).** Simply unzip the relevant `.zip` files after cloning.

For the data collection pipeline, please see [this companion GitHub repo](https://github.com/spfrommer/ranking_manipulation_data_pipeline).

# Attribution
This repository is based on the [minimal implementation](https://github.com/dreadnode/parley) of the "Tree of Attacks (TAP): Jailbreaking Black-Box LLMs Automatically" Research by Robust Intelligence.

[Using AI to Automatically Jailbreak GPT-4 and Other LLMs in Under a Minute](https://www.robustintelligence.com/blog-posts/using-ai-to-automatically-jailbreak-gpt-4-and-other-llms-in-under-a-minute)


# Usage
1. Install dependencies (inside a virtualenv)
```
pip install e .
```
2. Configure any required API keys
```
OPENAI_API_KEY='...'
TOGETHER_API_KEY='...'
PERPLEXITY_API_KEY='...'
```
3. (Optional) set up web server for `perplexity.ai` attack (commented out in `run.sh`). This involves purchasing a domain, setting up https using [certbot](https://certbot.eff.org/), selecting a password for `app.py` and `app_interface.py`, and running `run_server.sh` in `helpers` on the web server.
3. Reproduce results
```
bash scripts/run.sh
```
