import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()  # Load API Key
api_key = os.getenv("GOOGLE_API_KEY")

# Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=api_key
)

# App Title
st.title("🤖 Autonomous Data Science Co-Pilot")

st.write(
    "Upload CSV, Excel or JSON file and ask questions about your data."
)

# files
uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "json"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

    elif uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file)

    else:
        st.error("Unsupported file format")
        st.stop()

    # Preview of dataset
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    
    # Dataset Information
    st.subheader("📊 Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    # Automatic EDA
    st.subheader("🔍 Automatic Data Analysis")

    st.write("Column Data Types:")
    st.write(df.dtypes)

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    st.write("Statistical Summary:")
    st.write(df.describe())

    # Automatic Visualization
    st.subheader("📈 Automatic Visualizations")

    numeric_columns = df.select_dtypes(include=['number']).columns

    if len(numeric_columns) > 0:

        selected_column = st.selectbox(
            "Select column for visualization",
            numeric_columns
        )

        # Histogram
        st.write("Distribution Chart")

        fig, ax = plt.subplots()

        ax.hist(df[selected_column].dropna())

        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)


        # Correlation Heatmap
        if len(numeric_columns) > 1:

            st.write("Correlation Heatmap")

            fig, ax = plt.subplots()

            sns.heatmap(
                df[numeric_columns].corr(),
                annot=True,
                ax=ax
            )

            st.pyplot(fig)

    else:
        st.warning("No numerical columns available for visualization")
   # Automatic AI Insights

    st.subheader("🧠 AI Generated Insights")

    if st.button("Generate Dataset Insights"):

        insight_prompt = f"""
    You are an expert Data Scientist.

    Analyze this dataset briefly.

    Dataset columns:
    {list(df.columns)}

    Dataset summary:
    {df.describe().to_string()}

    Missing values:
    {df.isnull().sum().to_string()}

    Give ONLY:
    1. Three important insights
    2. Two data quality observations
    3. Two recommendations

    Keep the response under 150 words.
    Use simple bullet points only.
    """

        try:

            insight_response = llm.invoke(insight_prompt)

            if isinstance(insight_response.content, str):
                st.write(insight_response.content)

            elif isinstance(insight_response.content, list):

                for item in insight_response.content:

                    if isinstance(item, dict):

                        if "text" in item:
                            st.write(item["text"])

                        elif item.get("type") == "text":
                            st.write(item.get("text", ""))

            else:
                st.write(insight_response.content)

        except Exception as e:
            st.error(f"AI Error: {e}")
        st.subheader("🎯 Machine Learning Prediction")
    st.write("Choose the column you want to predict using Machine Learning.")

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    st.info(f"Selected Target: {target_column}")

    # ML Model Training

    if st.button("Train Model"):

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical columns
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(
            random_state=42
        )

        model.fit(X_train, y_train)
        # Feature Importance

        st.subheader("📌 Feature Importance")

        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        )

        st.dataframe(importance)

        fig, ax = plt.subplots()

        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")

        st.pyplot(fig)

        prediction = model.predict(X_test)

        r2 = r2_score(y_test, prediction)
        mae = mean_absolute_error(y_test, prediction)

        st.success("Model Training Completed ✅")

        st.write("R2 Score:", round(r2, 3))
        st.write("MAE:", round(mae, 3))
        # Prediction example
        st.subheader("Prediction Result")

        result = pd.DataFrame({
            "Actual": y_test,
            "Predicted": prediction
        })

        st.dataframe(result.head(10))
        st.subheader("🔮 Make New Prediction")

        input_data = {}

        for col in X.columns:
            input_data[col] = st.number_input(
                f"Enter {col}",
                value=0.0
            )

        if st.button("Predict Value"):

            input_df = pd.DataFrame(
                [input_data]
            )

            prediction_result = model.predict(input_df)

            st.success(
                f"Predicted {target_column}: {prediction_result[0]:.2f}"
            )
        st.subheader("📈 Actual vs Predicted")

        fig, ax = plt.subplots()

        ax.scatter(
            result["Actual"],
            result["Predicted"]
        )

        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Actual vs Predicted Comparison")

        st.pyplot(fig)
     
    # Question
    question = st.text_input(
        "Ask your data question"
    )

    # Analyze Button
    if st.button("Analyze"):

        if question:

            prompt = f"""
You are an expert Data Analyst.

Analyze this dataset.

Dataset columns:
{list(df.columns)}

First few rows:
{df.head().to_string()}

User Question:
{question}

Give a simple and professional answer with insights.
"""

            try:

                response = llm.invoke(prompt)

                st.subheader("🤖 AI Analysis")

                if isinstance(response.content, str):
                    st.write(response.content)

                elif isinstance(response.content, list):

                    for item in response.content:

                        if isinstance(item, dict):

                            if "text" in item:
                                st.write(item["text"])

                            elif item.get("type") == "text":
                                st.write(item.get("text", ""))

                else:
                    st.write(response.content)

            except Exception as e:

                st.error(f"AI Error: {e}")

        else:

            st.warning("Please enter a question.")