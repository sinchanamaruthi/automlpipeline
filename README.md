
# AutoML Pipeline — Phase 1 (Polished)

A clean Streamlit app that:
- Uploads CSV
- Auto-detects task (classification/regression)
- Preprocesses (impute + one-hot)
- Trains multiple models
- Auto-selects best
- Shows metrics, confusion matrix/ROC, feature importances
- Generates a simple natural-language summary

## How to run (locally)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud
- Push this repo to GitHub
- Create new app on Streamlit Cloud → point to `app.py`
- Done ✅

## Assumptions
- Last column is the target
- Binary ROC supported when there are 2 classes
