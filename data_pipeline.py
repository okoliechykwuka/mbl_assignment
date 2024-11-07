import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re
import logging
from typing import Tuple, Dict


class DataPreparationPipeline:
    def __init__(self, max_length: int = 512, model_name: str = "xlm-roberta-base"):
        """
        Initialize the data preparation pipeline.
        
        Args:
            max_length (int): Maximum sequence length for tokenization.
            model_name (str): Pretrained model name or path.
        """
        self.max_length = max_length
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoders = {}
        self.logger = self.setup_logger()
        self.model = None
        # To store the mapping for 'queue'
        self.queue_mapping = {np.int64(0): 'Billing and Payments', np.int64(1): 'Customer Service', np.int64(2): 'General Inquiry', np.int64(3): 'Human Resources', np.int64(4): 'IT Support', np.int64(5): 'Product Support', np.int64(6): 'Returns and Exchanges', np.int64(7): 'Sales and Pre-Sales', np.int64(8): 'Service Outages and Maintenance', np.int64(9): 'Technical Support'}
    def setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate the dataset.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated dataframe
        """
        try:
            df = pd.read_csv(file_path)
            required_columns = ['subject', 'body', 'queue', 'language', 'business_type']
            
            # Validate required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            self.logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Preprocess the dataset for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Preprocessed dataframe and preprocessing metadata
        """
        df_processed = df.copy()
        
        # Clean text fields
        df_processed['subject_cleaned'] = df_processed['subject'].apply(self.clean_text)
        df_processed['body_cleaned'] = df_processed['body'].apply(self.clean_text)
        
        # Combine subject and body for full text
        df_processed['full_text'] = df_processed['subject_cleaned'] + ' ' + df_processed['body_cleaned']
        
        # Drop rows where 'full_text' is NaN
        df_processed = df_processed.dropna(subset=['full_text'])  # Drop rows where 'full_text' is NaN
        
        # Convert full_text to string
        df_processed['full_text'] = df_processed['full_text'].astype(str) 
        
        # Encode categorical variables
        categorical_columns = ['queue', 'language', 'business_type']
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
            
        # Create preprocessing metadata
        preprocessing_meta = {
            'label_encoders': self.label_encoders,
            'num_classes': len(self.label_encoders['queue'].classes_),
            'language_distribution': df_processed['language'].value_counts().to_dict(),
            'queue_distribution': df_processed['queue'].value_counts().to_dict()
        }
        
        self.logger.info("Data preprocessing completed successfully")
        return df_processed, preprocessing_meta
        
    def prepare_datasets(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """
        Prepare train and test datasets for model training.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            Tuple[Dataset, Dataset]: Train and test datasets
        """
        
        # Encode target variable
        # df['queue_encoded'] = df['queue'].astype('category').cat.codes
        # queue_mapping = dict(enumerate(df['queue'].astype('category').cat.categories))
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df['full_text'], df['queue_encoded'], test_size=0.2, random_state=42)
        
        # Create Dataset objects for Hugging Face
        train_data = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
        test_data = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})
        
        # Tokenize data
        def preprocess_function(examples):
            return self.tokenizer(examples['text'], truncation=True, padding=True, max_length=self.max_length)
        
        train_data = train_data.map(preprocess_function, batched=True)
        test_data = test_data.map(preprocess_function, batched=True)
        
        return train_data, test_data
    
    # Compute metrics for evaluation
    def compute_metrics(self,p):
        preds = np.argmax(p.predictions, axis=1)
        return {
            'accuracy': (preds==p.label_ids).astype(np.float32).mean().item(),
        }
        
    def train_model(self, train_data: Dataset, test_data: Dataset, num_classes: int, output_dir: str = "./model"):
        """
        Train and save the model.
        
        Args:
            train_data (Dataset): Training dataset.
            test_data (Dataset): Testing dataset.
            output_dir (str): Directory to save the trained model.
        """
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_classes)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.logger.info("Model moved to GPU")
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=1,
            logging_dir='./logs',
            load_best_model_at_end=True,
        )
        
        
        # Define Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info(f"Model saved to {output_dir}")
        
    def load_model(self, model_dir: str = "./model"):
        """
        Load a saved model for inference.
        
        Args:
            model_dir (str): Path to the saved model directory.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.logger.info(f"Model loaded from {model_dir}")

