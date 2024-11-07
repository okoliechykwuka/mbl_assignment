from data_pipeline import DataPreparationPipeline


# Example usage:
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPreparationPipeline()

    # Load and preprocess data
    df = pipeline.load_data("dataset/support_tickets_preprocessed.csv")
    df_processed, preprocessing_meta = pipeline.preprocess_data(df)
    train_data, test_data = pipeline.prepare_datasets(df_processed)
    # Train and save the model using num_classes from preprocessing metadata
    pipeline.train_model(train_data, test_data, num_classes=preprocessing_meta['num_classes'], output_dir="./model")