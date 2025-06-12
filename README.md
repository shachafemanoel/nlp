# Tweet Sentiment Extraction

## Project Overview

This repository implements a solution for the Tweet Sentiment Extraction challenge. It extracts from each tweet the exact words or phrase that express its sentiment label. Sentiment labels include positive, negative or neutral.

## Key Features

1. Fine-tuned BERT-base model for span extraction
2. Custom preprocessing pipeline including tokenization and text cleaning
3. Training scripts with hyperparameter tuning and learning rate scheduling
4. Evaluation using Jaccard score and visual diagnostics
5. Sample inference script for real-time predictions

## Repository Structure

1. `data/`

   * raw and processed CSV files for train, validation and test sets
2. `notebooks/`

   * exploratory data analysis and visualization notebooks
3. `src/`

   * `preprocessing.py` functions for text cleaning and tokenization
   * `model.py` BERT span extraction model definition
   * `train.py` training and validation loop
   * `predict.py` inference pipeline
4. `results/`

   * plots and sample prediction tables
5. `configs/`

   * YAML files for experiment configurations
6. `README.md` this documentation

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/tweet-sentiment-extraction.git
   cd tweet-sentiment-extraction
   ```
2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. Download the Kaggle dataset and place files in `data/` directory.
2. Preprocess data:

   ```bash
   python src/preprocessing.py --input data/train.csv --output data/train_cleaned.csv
   ```
3. Train the model:

   ```bash
   python src/train.py --config configs/bert_config.yaml
   ```
4. Evaluate on validation set:

   ```bash
   python src/predict.py --model checkpoints/best_model.pt --input data/val.csv --output results/predictions.csv
   ```
5. View results and plots in `results/` directory.

## Training Details

Training uses AdamW optimizer with linear warm-up and decay. We train for 3 epochs with batch size 16 and learning rate 3e-5. Early stopping on validation Jaccard score is applied.

## Evaluation

We measure performance using the Jaccard similarity metric. Sample Jaccard scores and loss curves are available in `results/`.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with detailed description of changes.

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback please reach out at [your.email@example.com](mailto:your.email@example.com).
