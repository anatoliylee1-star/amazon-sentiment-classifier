# src/inference.py
from typing import Optional, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentService:
    """
    Inference wrapper for sentiment classification.

    Usage:
      # Pretrained, no fine-tuning (instant demo):
      svc = SentimentService()  # uses SST-2 model from Hugging Face

      # Your fine-tuned model later:
      svc = SentimentService(model_dir="models/distilbert_best")
    """
    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None):
        # If no local fine-tuned model dir is provided, use a strong pretrained SST-2 model
        # You can swap this to your own Hugging Face repo if you host the fine-tuned model there.
        hf_model = "distilbert-base-uncased-finetuned-sst-2-english" if model_dir is None else model_dir

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(hf_model)

        # CPU by default (safe). If you have a GPU, set device="cuda"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Label resolution: be robust to either id-based or string labels
        # For SST-2: id 0 -> NEGATIVE, id 1 -> POSITIVE
        # For your fine-tuned model, config may have id2label mapping.
        self.id2label = getattr(self.model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})

    @torch.inference_mode()
    def predict(self, text: str, max_length: int = 256) -> Dict[str, float | str]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        ).to(self.device)
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        label_id = int(torch.argmax(probs).item())
        label_name = self.id2label.get(label_id, str(label_id)).upper()

        # Normalize to the exact strings your UI expects:
        if "POS" in label_name:
            ui_label = "Positive review"
        elif "NEG" in label_name:
            ui_label = "Negative review"
        else:
            # Fallback if a custom label sneaks in
            ui_label = label_name.title()

        return {
            "label": ui_label,
            "confidence": round(float(probs[label_id].item()), 4)
        }

