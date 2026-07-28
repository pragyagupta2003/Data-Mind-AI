# Autonomous Data Science Co-Pilot

An intelligent Data Science web application built using **Python**, **Streamlit**, **Machine Learning**, and **Google Gemini AI**. This project allows users to upload datasets, perform exploratory data analysis, visualize data, train a machine learning model, generate predictions, and ask natural language questions about their data.

The goal of this project is to simplify data analysis by combining traditional data science techniques with Generative AI, making it useful for students, beginners, and anyone working with datasets.

# Features
* Upload datasets in CSV, Excel, or JSON format
* Preview uploaded data
* Display dataset information such as rows, columns, and missing values
* Perform automatic exploratory data analysis (EDA)
* Generate descriptive statistics
* Visualize numerical data using histograms
* Display correlation heatmaps
* Train a Random Forest Regression model
* Show model performance using R² Score and Mean Absolute Error (MAE)
* Display feature importance
* Compare actual and predicted values
* Predict values for new user inputs
* Ask questions about the dataset using Google Gemini AI
  
# Technologies Used
* Python
* Streamlit
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* LangChain
* Google Gemini API
* Python-dotenv
* OpenPyXL

# Project Structure
Autonomous-Data-Science-CoPilot/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env

# How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
```

### 2. Open the project folder

```bash
cd your-repository
```

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Add your Google Gemini API key inside the `.env` file.

```text
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

# Machine Learning Workflow
1. Upload a dataset.
2. Select the target column.
3. Convert categorical columns using one-hot encoding.
4. Split the dataset into training and testing sets.
5. Train a Random Forest Regressor.
6. Evaluate the model using R² Score and MAE.
7. Display feature importance.
8. Predict values for unseen data.
9. Compare actual and predicted values visually.

# AI Capabilities

The application uses Google Gemini AI to answer questions related to the uploaded dataset. Users can type questions in natural language and receive meaningful insights generated from the data.

Example questions:

* Which feature has the highest impact?
* What trends are visible in this dataset?
* Are there any missing values?
* Which columns are most correlated?
* What business insights can be derived?

# Supported File Formats

* CSV (.csv)
* Excel (.xlsx)
* JSON (.json)

# Future Improvements

Some additional features that can be added in future versions include:

* Classification models
* Multiple algorithm selection
* Hyperparameter tuning
* Model comparison dashboard
* Download prediction results
* Interactive Plotly visualizations
* Automatic PDF report generation
* Time-series forecasting support
* Model saving and loading
* Feature selection techniques
* 
# Learning Outcomes
Through this project I gained practical experience in:

* Data preprocessing
* Exploratory Data Analysis (EDA)
* Data visualization
* Machine Learning model development
* Model evaluation
* Feature engineering
* Streamlit application development
* Integration of Google Gemini AI using LangChain
* Building an end-to-end Data Science application

# Author

**Pragya Gupta**

MCA Final Year Student

Interested in Data Science, Artificial Intelligence, Machine Learning, and Generative AI.

This project was developed as a hands-on learning project to explore how AI and Machine Learning can be integrated into an interactive data analysis application.
## Live Demo

Streamlit App:
https://cei-internship-jkbcx9pdrzoxzqrfgvrnkh.streamlit.app/
