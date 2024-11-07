import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import  Dict
from data_pipeline import DataPreparationPipeline


class Predictor(DataPreparationPipeline):
    def __init__(self):
        """
        Initialize the data preparation pipeline.
        
        Args:
            max_length (int): Maximum sequence length for tokenization.
            model_name (str): Pretrained model name or path.
        """
        super().__init__()
        

    def load_model(self, model_dir: str = "./model"):
        """
        Load a saved model for inference.
        
        Args:
            model_dir (str): Path to the saved model directory.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()  # Set model to evaluation mode
        self.logger.info(f"Model loaded from {model_dir}")

    def predict(self, text: str) -> Dict:
        """
        Predict the label for a given text sample.
        
        Args:
            text (str): Input text to classify.
        
        Returns:
            Dict: Predicted label and confidence score.
        """
        # Preprocess the text
        processed_text = self.clean_text(text)
        
        # Tokenize the text
        inputs = self.tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        
        # Move tensors to GPU if available
        if torch.cuda.is_available():
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            self.model.to("cuda")
        
        # Generate predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
        # Map prediction index to label
        label = self.queue_mapping[predicted_class_idx]
        
        return {"label": label, "confidence score": confidence}
