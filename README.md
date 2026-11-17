
# 🚀 Social Media Virality Prediction Using Machine Learning

### Predicting the Viral Potential of Social Media Posts Using Machine Learning

## 📌 Project Overview

Social Media Virality Predictor is a Machine Learning-based web application that predicts the viral potential of social media posts and reels based on their engagement metrics.

Users can manually enter the number of views, likes, comments, and shares. The trained Machine Learning model analyzes these metrics and predicts a virality score between 0% and 100%.

The project is developed using Python, Scikit-learn, and Streamlit.

## 🎯 Objectives

- Predict the virality potential of social media posts.
- Analyze social media engagement metrics.
- Calculate engagement rates and interaction ratios.
- Classify posts based on their predicted virality score.
- Provide an interactive dashboard for prediction and analysis.

## ✨ Features

- Manual input of views, likes, comments, and shares.
- Virality score prediction (0–100%).
- Virality category classification.
- Engagement rate calculation.
- Like-to-view ratio calculation.
- Comment-to-view ratio calculation.
- Share-to-view ratio calculation.
- Interactive Streamlit dashboard.
- Engagement breakdown visualization.
- Machine Learning-based prediction.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computation |
| Scikit-learn | Machine Learning |
| Random Forest Regressor | Prediction Model |
| Streamlit | Web Application |
| Matplotlib | Data Visualization |
| Joblib | Model Saving and Loading |

## 🤖 Machine Learning Algorithm

This project uses **Random Forest Regression**, a supervised Machine Learning algorithm.

The model analyzes engagement metrics and predicts a virality score.

### Input Features

- Views
- Likes
- Comments
- Shares
- Like/View Ratio
- Comment/View Ratio
- Share/View Ratio
- Engagement Rate

### Virality Categories

| Virality Score | Category |
|---|---|
| 0–20% | Low |
| 21–40% | Moderate |
| 41–60% | Good |
| 61–80% | High |
| 81–100% | Viral |

## 📊 Model Evaluation

The model is evaluated using the following regression metrics:

| Evaluation Metric | Score |
|---|---:|
| MAE | 1.7708 |
| MSE | 9.1862 |
| RMSE | 3.0309 |
| R² Score | 0.959 |

**Note:** These evaluation results are based on a synthetic dataset generated for demonstration purposes. They do not represent guaranteed real-world prediction accuracy.

## 📁 Project Structure

```text
Social-Media-Virality-Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── social_media_data.csv
│
└── models/
    ├── virality_model.pkl
    ├── feature_importance.png
    └── training_metrics.json
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Social-Media-Virality-Prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd Social-Media-Virality-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Train the Machine Learning Model

```bash
python train_model.py
```

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser:

```text
http://localhost:8501
```

## 📈 How It Works

1. Open the application.
2. Enter the number of views.
3. Enter the number of likes.
4. Enter the number of comments.
5. Enter the number of shares.
6. Click on the **Predict Virality** button.
7. The Machine Learning model processes the input.
8. View the predicted virality score and category.
9. Analyze engagement rates and visualizations.

## 📚 Dataset Information

The project uses a synthetic social media dataset generated using Python and NumPy.

The dataset contains sample social media posts with engagement-related features.

The dataset is intended for demonstration and educational purposes.

For better real-world predictions, the model should be trained using actual historical social media engagement data.

## 🚀 Future Enhancements

- Add repost metrics as an additional input.
- Integrate Instagram and YouTube APIs.
- Use real-world social media datasets.
- Compare different Machine Learning algorithms.
- Add historical prediction tracking.
- Develop advanced engagement analytics.
- Deploy the application online.

## 💡 Applications

- Social Media Analytics
- Digital Marketing
- Content Performance Analysis
- Influencer Marketing
- Audience Engagement Analysis
- Viral Content Prediction

## 👨‍💻 Project Information

**Project Name:** Social Media Virality Predictor

**Domain:** Data Science and Machine Learning

**Algorithm:** Random Forest Regression

**Frontend:** Streamlit

**Programming Language:** Python

**Model Library:** Scikit-learn

---

## 👨‍💻 Author

**Manav Natekar**
