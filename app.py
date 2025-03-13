import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.datasets import load_iris

# --- Configuration and Setup ---

# Load environment variables from .env file
load_dotenv()

# Get environment variables with default values.
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
API_BASE_URL = f"http://fastapi:{FASTAPI_PORT}"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
DATABASE_FOLDER = os.getenv("DATABASE_FOLDER", "db")
DATABASE = os.path.join(DATABASE_FOLDER, "models.db")

# region Frontend Functions
# --- Frontend-Backend Interaction Functions ---
# These functions manage the communication between the Streamlit frontend and the FastAPI backend.


def clear_models():
    """Clears only the model files (pickle files) inside the models directory.

    This function removes .pkl files from the MODELS_DIR. It checks for the
    existence of the directory and handles potential errors gracefully.
    It returns a status message and type.
    """
    if not os.path.exists(MODELS_DIR):
        return "Models directory does not exist or is already empty.", "warning"

    try:
        for filename in os.listdir(MODELS_DIR):
            file_path = os.path.join(MODELS_DIR, filename)
            if filename.endswith(".pkl") and os.path.isfile(file_path):
                os.remove(file_path)

        return "Models (pickle files) cleared.", "success"

    except OSError as e:
        # Log the error more verbosely for debugging.
        print(f"ERROR: Failed to clear models directory: {e}")
        print(
            f"ERROR: Current contents of MODELS_DIR: {os.listdir(MODELS_DIR)}"
        ) if os.path.exists(MODELS_DIR) else print("ERROR: MODELS_DIR does not exist")
        return f"Error clearing models: {e}. Please check the logs.", "error"


def train_model(dataset, model_name, hyperparameters):
    """Trains a model via the FastAPI backend.

    This function sends a request to the /train endpoint of the FastAPI
    application to train a model. It handles the request and response,
    updates the session state, and manages potential errors.

    Args:
        dataset (pd.DataFrame): The dataset used for training.
        model_name (str): The name of the model to train.
        hyperparameters (dict): The hyperparameters for the model.

    Returns:
        tuple: A tuple containing the model_id and the full model_info dictionary,
                or (None, None) if an error occurred.
    """
    dataset_json = dataset.to_json(orient="split")

    try:
        payload = {
            "dataset": dataset_json,
            "model_name": model_name,
            "hyperparameters": hyperparameters,
        }

        response = requests.post(f"{API_BASE_URL}/train", json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        model_info = response.json()

        # Validate the response format.
        if not isinstance(model_info, dict):
            st.session_state["model_message"] = (
                f"Unexpected response format: {model_info}"
            )
            st.session_state["model_message_type"] = "error"
            return None, None

        # Check if the model already exists (fastapi side message).
        if "message" in model_info:
            st.session_state["model_message"] = model_info["message"]
            st.session_state["model_message_type"] = "warning"
        else:
            st.session_state["model_message"] = "Model trained successfully."
            st.session_state["model_message_type"] = "success"

        # Update session state with model info.
        st.session_state["model_id"] = model_info["model_id"]
        st.session_state["confusion_matrix"] = model_info.get("confusion_matrix")
        st.session_state["classification_report"] = model_info.get(
            "classification_report"
        )
        st.session_state["f1_score"] = model_info.get("f1_score")

        return model_info["model_id"], model_info

    except requests.exceptions.RequestException as e:
        st.session_state["model_message"] = f"Error during model training: {e}"
        st.session_state["model_message_type"] = "error"
        return None, None


def make_prediction(model_id, input_data, dataset_choice):
    """
    Sends a prediction request to the FastAPI backend.

    Args:
        model_id (str): The ID of the model to use for prediction.
        input_data (list): The input data for the prediction.
        dataset_choice (str): The dataset used for training ("Iris Dataset" or "Upload a CSV file")

    Returns:
        dict: The prediction result, or None if an error occurs.
    """
    try:
        payload = {"model_id": model_id, "input_data": input_data}

        response = requests.post(f"{API_BASE_URL}/predict", json=payload)
        response.raise_for_status()
        prediction_result = response.json()
        # Enhance prediction result for Iris dataset
        if dataset_choice == "Iris Dataset":
            iris_target_names = {
                0: "setosa",
                1: "versicolor",
                2: "virginica",
            }
            predicted_class = prediction_result["prediction"]
            prediction_result["prediction_name"] = iris_target_names.get(
                predicted_class, "Unknown"
            )

        return prediction_result
    except requests.exceptions.RequestException as e:
        st.error(f"Error during prediction: {e}")
        return None


# endregion

# region Page Settings, Title, and Cleanup Button
# --- Streamlit Application ---
st.set_page_config(layout="wide", page_title="AIDI Project", page_icon=":computer:")


# Sidebar Cleanup
st.sidebar.subheader("Cleanup")
clear_button = st.sidebar.button("Clear Models")
# Display message under the Clear Models Button
if "clear_message" not in st.session_state:
    st.session_state["clear_message"] = ""
    st.session_state["clear_message_type"] = ""

if clear_button:
    # Execute the clear_models function
    message, message_type = clear_models()

    # Overwrite the message with the result of the operation
    st.session_state["clear_message"] = message
    st.session_state["clear_message_type"] = message_type

    # Display the message
    if st.session_state["clear_message_type"] == "error":
        st.sidebar.error(st.session_state["clear_message"], icon="🚨")
    elif st.session_state["clear_message_type"] == "success":
        st.sidebar.success(st.session_state["clear_message"], icon="✅")


st.title("André Lima's ML AIDI Project")
st.write(
    """
    - app.py (Frontend - Streamlit): handles the frontend of the application, providing 
    the user interface for data selection, model selection, hyperparameter tuning, and 
    displaying results.
    - main.py (Backend - FastAPI): defines the API endpoints for data storage, model 
    training, and prediction. It handles the logic for interacting with the database 
    and training machine learning models.
    - database.py (Backend - Database): contains the code for database setup, model 
    definition, and database interaction functions. It provides an abstraction layer 
    for working with the database, making it easier to manage and maintain.
    - Framework is containerized using Docker Containers.
    ---
    """
)

# Wait until FastAPI is available
st.write("Waiting for FastAPI to be ready...")
max_retries = 10
retries = 0
while retries < max_retries:
    try:
        requests.get(f"{API_BASE_URL}/health")
        st.write("FastAPI is ready!")
        break
    except requests.exceptions.ConnectionError:
        retries += 1
        time.sleep(1)
        st.write(f"FastAPI is not ready yet, retrying... ({retries}/{max_retries})")
if retries == max_retries:
    st.error("Failed to connect to FastAPI after multiple retries.")
# endregion

# region Data Selection
# --- Session State Initialization ---
# Initialize session state variables to manage the application state across interactions.
if "dataset_choice" not in st.session_state:
    st.session_state.dataset_choice = None
if "selected_data" not in st.session_state:
    st.session_state.selected_data = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "model_message" not in st.session_state:
    st.session_state["model_message"] = ""
    st.session_state["model_message_type"] = ""
if "model_id" not in st.session_state:
    st.session_state["model_id"] = None
if "confusion_matrix" not in st.session_state:
    st.session_state["confusion_matrix"] = None
if "classification_report" not in st.session_state:
    st.session_state["classification_report"] = None
if "f1_score" not in st.session_state:
    st.session_state["f1_score"] = None

# --- Dataset Selection Section ---
st.subheader("1 - Choose a Dataset:")
dataset_choice = st.radio(
    "Select Dataset",
    ["Iris Dataset", "Upload a CSV file"],
    index=0
    if st.session_state.dataset_choice == "Iris Dataset"
    else (1 if st.session_state.dataset_choice == "Upload a CSV file" else None),
    horizontal=True,
    key="dataset_choice_radio",
)

# Reset session state related to data when the dataset_choice changes.
if dataset_choice != st.session_state.dataset_choice:
    st.session_state.dataset_choice = dataset_choice
    st.session_state.selected_data = None
    st.session_state["model_message"] = ""
    st.session_state["model_message_type"] = ""
    st.session_state["model_id"] = None
    st.session_state["confusion_matrix"] = None
    st.session_state["classification_report"] = None
    st.session_state["f1_score"] = None


# Handle Iris Dataset selection.
if dataset_choice == "Iris Dataset":
    if (
        st.session_state.dataset_choice != "Iris Dataset"
        or st.session_state.selected_data is None
    ):
        iris = load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        data["target"] = iris.target
        cols = ["target"] + iris.feature_names
        data = data[cols]
        st.session_state.selected_data = data
        st.session_state.dataset_choice = "Iris Dataset"
    if st.session_state.selected_data is not None:
        st.dataframe(st.session_state.selected_data)

# Handle CSV file upload.
elif dataset_choice == "Upload a CSV file":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            # check if the target column exists
            if "target" not in [col.lower() for col in data.columns]:
                st.error(
                    "The uploaded dataset is invalid. It must have a column named 'target'.",
                    icon="🚨",
                )
                st.session_state.selected_data = None
                st.session_state.uploaded_file_name = None
            else:
                # search for the correct case for the column 'target'
                for col in data.columns:
                    if col.lower() == "target":
                        data = data.rename(columns={col: "target"})
                        break
                st.session_state.selected_data = data
                st.session_state.uploaded_file_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading CSV: {e}", icon="🚨")
            st.session_state.selected_data = None
            st.session_state.uploaded_file_name = None

        if st.session_state.selected_data is not None:
            st.dataframe(st.session_state.selected_data)

    else:
        st.info(
            "Please upload a CSV file. The dataset must have a column named 'target'.",
            icon="ℹ️",
        )
st.divider()
# endregion

# region Model Selection
# --- Model Selection Section ---
st.subheader("2 - Choose a Model and it's Hyperparameters: ")
model_name = st.selectbox(
    "Select Model",
    ["kNN - k-Nearest Neighbors", "SVM - Support Vector Machine", "DT - Decision Tree"],
    key="model_name",
)

# Store hyperparameters in this dictionary.
hyperparameter_values = {}

# Model-specific hyperparameter selection.
if model_name == "kNN - k-Nearest Neighbors":
    hyperparameter_values["k"] = st.slider("k (number of neighbors)", 1, 10, 3, key="k")
    hyperparameter_values["weights"] = st.selectbox(
        "Weights", ["uniform", "distance"], key="weights"
    )
    # Clear session state if previously set by another model type.
    if "C" in st.session_state:
        del st.session_state["C"]
    if "max_depth" in st.session_state:
        del st.session_state["max_depth"]
elif model_name == "SVM - Support Vector Machine":
    hyperparameter_values["C"] = st.slider(
        "C (Regularization parameter)", 0.1, 10.0, 1.0, key="C"
    )
    hyperparameter_values["kernel"] = st.selectbox(
        "Kernel", ["linear", "poly", "rbf"], key="kernel"
    )
    if "k" in st.session_state:
        del st.session_state["k"]
    if "max_depth" in st.session_state:
        del st.session_state["max_depth"]
elif model_name == "DT - Decision Tree":
    hyperparameter_values["max_depth"] = st.slider(
        "Max Depth", 1, 10, 3, key="max_depth"
    )
    hyperparameter_values["criterion"] = st.selectbox(
        "Criterion", ["gini", "entropy"], key="criterion"
    )
    if "k" in st.session_state:
        del st.session_state["k"]
    if "C" in st.session_state:
        del st.session_state["C"]
st.divider()
# endregion

# region Train Model
# --- Submit Data for Model Training ---
st.subheader("3 - Train the Model and check it's metrics: ")
if st.button("Train Model", key="train_button"):
    if st.session_state.selected_data is None:
        st.error("Please select or upload a dataset first.")
    else:
        with st.spinner("Training model..."):
            model_id, model_info = train_model(
                st.session_state.selected_data, model_name, hyperparameter_values
            )
            # Handle model training response and display.
            if st.session_state["model_message_type"] == "error":
                st.error(st.session_state["model_message"], icon="🚨")
            elif st.session_state["model_message_type"] == "warning":
                st.warning(st.session_state["model_message"], icon="⚠️")
            elif st.session_state["model_message_type"] == "success":
                st.success(st.session_state["model_message"], icon="✅")

            # Display model metrics if the model was trained successfully.
            if model_id:
                st.write(f"**Model ID:** {st.session_state['model_id']}")

                if st.session_state["f1_score"] is not None:
                    st.write(f"**F1-Score:** {st.session_state['f1_score']:.2%}")

                if st.session_state["confusion_matrix"] is not None:
                    st.write("**Confusion Matrix:**")
                    conf_matrix_df = pd.DataFrame(st.session_state["confusion_matrix"])
                    st.dataframe(conf_matrix_df)

                if st.session_state["classification_report"] is not None:
                    st.write("**Classification Report:**")
                    report = st.session_state["classification_report"]
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
            else:
                st.write("Something went wrong during model training.")
st.divider()
# endregion

# region Predict
# --- Make a Prediction ---
st.subheader("4 - Make a Prediction:")
if "model_id" not in st.session_state:
    st.write("Train a model before trying to predict.")
else:
    st.write(f"Model ID: {st.session_state.model_id}")

    if st.session_state.selected_data is not None:
        feature_names = [
            col for col in st.session_state.selected_data.columns if col != "target"
        ]
        min_max_values = {}
        for feature in feature_names:
            min_max_values[feature] = (
                st.session_state.selected_data[feature].min(),
                st.session_state.selected_data[feature].max(),
            )
    else:
        feature_names = []
        min_max_values = {}

    input_values = []
    for feature_name in feature_names:
        if feature_name in min_max_values:
            min_val, max_val = min_max_values[feature_name]
            label = f"{feature_name} (min: {min_val:.2f}, max: {max_val:.2f})"
        else:
            label = feature_name
        input_values.append(st.number_input(label, key=f"feature_{feature_name}"))

    if st.button("Make Prediction", key="predict_button"):
        with st.spinner("Predicting..."):
            try:
                prediction_result = make_prediction(
                    st.session_state.model_id,
                    input_values,
                    st.session_state.dataset_choice,
                )
                if prediction_result:
                    if "prediction_name" in prediction_result:
                        st.success(
                            f"Prediction: {prediction_result['prediction']} ({prediction_result['prediction_name']})"
                        )
                    else:
                        st.success(f"Prediction: {prediction_result['prediction']}")
                else:
                    st.error("Error: Prediction result is None.")

            except requests.exceptions.RequestException as e:
                st.error(f"Error during prediction: {e}")

st.divider()
# endregion
