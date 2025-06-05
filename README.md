# Climate Sentiment Analysis using Streamlit


## Project Overview
This project uses machine learning to classify climate change-related tweets into sentiment categories. It allows users interested in sustainable markets to understand public opinion about climate change. The trained classification model is deployed via a Streamlit app, making real-time sentiment analysis easily accessible.

## Project structure

**Classification notebook folder**

Contains the main notebook responsible for:
- Loading the dataset
- Preprocessing the data
- Training and evaluating the classification model

**Streamlit app folder**

Includes:
- The base code for the Streamlit web application
- A saved model used by the app
- Any additional resources required to run the app

## How to Run the App Locally

Follow these steps by running the given commands within a Git bash (Windows), or terminal (Mac/Linux) to run the Streamlit app on your local machine:

1. Install Required Libraries

Make sure Python and the required libraries are installed:

`pip install -U streamlit numpy pandas scikit-learn textblob`

**Note:**
Since our preprocessing or model relies on NLTK stopwords and punctuation handling, you’ll need to download the required NLTK resources. Run the following in a Python shell or add it to your script:

```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
2. Clone the Repository

`git clone https://github.com/MaureenMbugua/Climate-Sentiment-Analysis-streamlit.git`

3. Navigate to the Streamlit App Folder

`cd Streamlit_app`

4. Run the Streamlit App

`streamlit run base_app.py`

If successful, you’ll see a message like:

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501

Your default browser should automatically open the app at http://localhost:8501.

## Next Steps & Recommendations

**- Model Improvements:**

The current model used in the app is a baseline. We should replace it with the highest-performing version from the classification notebook for improved accuracy.

**- Cloud Deployment:**

Hosting the app on a cloud server (like AWS EC2) allows anyone to access it without running commands locally.
