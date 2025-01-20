# Sentiment Analysis Tool for Product Reviews

## Overview

This project implements a sentiment analysis tool using Python and scikit-learn to classify product reviews into positive, neutral, or negative categories. The tool preprocesses text data, utilizes TF-IDF for feature extraction, and employs logistic regression for classification.

## Features

- Text preprocessing using NLTK
- TF-IDF vectorization for feature extraction
- Logistic regression for sentiment classification
- Support for training on the Amazon Fine Food Reviews dataset
- Evaluation of model performance
- Prediction of sentiment for new reviews

## Requirements

- Python 3.7+
- pandas
- scikit-learn
- nltk
- Kaggle account (for downloading the dataset)

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

## Dataset

This project uses the Amazon Fine Food Reviews dataset. To download the dataset:

1. Go to the Kaggle dataset page: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. If you don't have a Kaggle account, you'll need to create one (it's free).
3. Click the "Download" button on the dataset page.
4. Once downloaded, extract the ZIP file.
5. Locate the file named "Reviews.csv" in the extracted folder.
6. Move "Reviews.csv" to the same directory as the sentiment analysis script.

## Usage

1. Ensure you've downloaded and placed the "Reviews.csv" file in the project directory as described in the Dataset section.

2. Run the script:
   ```
   python main.py
   ```

3. The script will output:
   - A classification report showing the model's performance
   - Predictions for a few example reviews

## Customization

- You can adjust the `max_features` parameter in the `TfidfVectorizer` to change the number of features used.
- The `classify_sentiment` function can be modified to change how numerical scores are mapped to sentiment categories.

## Contributing

Contributions to improve the tool are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For any questions or feedback, please open an issue in the GitHub repository.
