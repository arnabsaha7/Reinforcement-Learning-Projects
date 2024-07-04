from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if news:
        tokens = tokenizer(news, padding=True, truncation=True, return_tensors="pt").to(device)
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

        # Sum the logits across the batch dimension (if needed)
        result_sum = torch.sum(result, dim=0)
        
        # Apply softmax to the summed logits
        probability = torch.nn.functional.softmax(result_sum, dim=0)
        
        # Get the maximum probability and corresponding sentiment
        max_prob = torch.max(probability)
        sentiment = labels[torch.argmax(probability)]
        
        return max_prob.item(), sentiment
    else:
        return 0, labels[-1]

if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(["markets responded positively to the scenario"])
    print(tensor, sentiment)
    print(torch.cuda.is_available())
