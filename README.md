# Sentiment Analysis Tool for Product Reviews

## Overview

This project implements a sentiment analysis tool using Python and scikit-learn to classify product reviews into positive, neutral, or negative categories. The tool preprocesses text data, utilizes TF-IDF for feature extraction, and employs logistic regression for classification.

## Features

- Text preprocessing using NLTK
- TF-IDF vectorization for feature extraction
- Logistic regression for sentiment classification
- Support for training on custom datasets
- Evaluation of model performance
- Prediction of sentiment for new reviews

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/sentiment-analysis-tool.git
   cd sentiment-analysis-tool
   ```

2. Install the required packages:
   ```
   pip install pandas scikit-learn nltk
   ```

3. Download the required NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   nltk.download('punkt_tab')
   ```

## Usage

1. Prepare your dataset:
   - The script expects a CSV file named 'Reviews.csv' in the same directory.
   - The CSV should have at least two columns: 'Text' (containing the review text) and 'Score' (containing the numerical rating).

2. Run the script:
   ```
   python sentiment_analysis.py
   ```

3. The script will output:
   - A classification report showing the model's performance
   - Predictions for a few example reviews

## Customization

- To use a different dataset, modify the `load_data` function in the script to match your CSV file's structure.
- You can adjust the `max_features` parameter in the `TfidfVectorizer` to change the number of features used.
- The `classify_sentiment` function can be modified to change how numerical scores are mapped to sentiment categories.

## Contributing

Contributions to improve the tool are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For any questions or feedback, please open an issue in the GitHub repository.
