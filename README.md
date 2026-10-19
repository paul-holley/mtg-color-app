# MTG Color Classifier

A web app: upload a picture and a neural network trained on Magic: The Gathering
card art guesses which of the five Magic colors it belongs to (white, blue, black,
red, green). It can pick several colors, or none.

## How it works

- The model is a fine-tuned ConvNeXt image classifier with five independent outputs,
  one per color, each with its own decision threshold. It only ever sees the image.
- The trained model is hosted on Hugging Face:
  [paularoni/mtg-color-classifier](https://huggingface.co/paularoni/mtg-color-classifier).
  The model card there lists the current model and its test scores.
- The app downloads that model once when it starts. A better model can be published
  there at any time; the app uses it after its next restart, with no code change.
- Uploaded images are processed in memory and never saved or logged.

## Run locally

```
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Unofficial fan project, not affiliated with or endorsed by Wizards of the Coast.
Card art is © Wizards of the Coast and its artists; training data from Scryfall.
