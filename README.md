# Motorcycle Value Estimator

A motorcycle value estimation app that combines Computer Vision, Machine Learning, and NLP.

## How it works

1. **Computer Vision** – Upload a motorcycle photo, a fine-tuned ViT model recognizes the brand
2. **ML Numeric** – A Gradient Boosting model predicts the market value based on brand, year, km, and owner
3. **NLP** – An LLM generates a German explanation of the price estimate

## Deployment

The app is deployed on Hugging Face Spaces:
https://huggingface.co/spaces/durovali/motorcycle-value-estimator

## Data Sources

- [Kaggle Motorcycle Dataset](https://www.kaggle.com/datasets/nehalbirla/motorcycle-dataset) – 1061 used motorcycle listings (structured CSV)
- [Custom-collected motorcycle images](https://huggingface.co/durovali/vit-motorcycle) – 55 images across 6 brands (BMW, Honda, Kawasaki, Suzuki, Triumph, Yamaha) 

## Project Files

| File | Description |
|------|-------------|
| `app.py` | Main application (Gradio interface) |
| `bike_details_training.ipynb` | ML model training notebook |
| `motorcycle-dataset.ipynb` | CV model training notebook |
| `motorcycle_price_model.pkl` | Trained Gradient Boosting model |
| `brand_codes.pkl` | Brand encoding mapping |
| `BIKE DETAILS.csv` | Kaggle motorcycle dataset |
| `documentation.md` | Full project documentation |
| `requirements.txt` | Python dependencies |

## How to Run

```bash
pip install gradio openai transformers torch numpy scikit-learn==1.6.1
export OPENAI_API_KEY="your-key"
python app.py
```

## Documentation

See [documentation.md](documentation.md) for the full project documentation.
