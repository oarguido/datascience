import os
import pickle
import uuid
from contextlib import asynccontextmanager
from io import StringIO

import pandas as pd
from database import create_model, create_prediction, create_table, get_model
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# --- Configuration and Setup ---

# Load environment variables from the .env file.
load_dotenv()

# Retrieve the directories for storing models and database from environment variables.
# If not set, use default values.
MODELS_DIR = os.getenv("MODELS_DIR", "models")
DATABASE_FOLDER = os.getenv("DATABASE_FOLDER", "db")

# --- Directory Handling ---

# Create the models directory if it doesn't exist.
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)


# --- FastAPI Lifespan Events ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Asynchronous context manager for FastAPI application startup and shutdown events.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Code to run on startup
    create_table()  # Create the database tables.
    yield  # Yield control to the application.
    # Code to run on shutdown (currently empty).


# --- FastAPI Application ---

# Create the FastAPI application instance with the lifespan event handlers.
app = FastAPI(lifespan=lifespan)

# Dictionary to store trained models (in-memory for this example).
# NOTE: In a production setting, consider using a persistent storage solution.
trained_models = {}


# --- API Endpoints ---


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the API is running.

    Returns:
        dict: A dictionary with the status "ok".
    """
    return {"status": "ok"}


@app.get("/")
async def root():
    """
    Root endpoint to provide a basic welcome message.

    Returns:
        dict: A dictionary with a welcome message.
    """
    return {"message": "André Lima ML FastAPI (backend)!"}


@app.post("/train")
async def train_model(request_data: dict):
    """
    Endpoint to train a machine learning model based on the provided data.

    Args:
        request_data (dict): A dictionary containing the training data, model name, and hyperparameters.
                            Example:
                            {
                                "dataset": "JSON string representing the dataset",
                                "model_name": "kNN - k-Nearest Neighbors",
                                "hyperparameters": {"k": 3, "weights": "uniform"}
                            }

    Returns:
        dict: A dictionary containing the model ID, accuracy, confusion matrix, and classification report.
            Example:
            {
                "model_id": "some-uuid",
                "accuracy": 0.95,
                "confusion_matrix": [[10, 0], [1, 9]],
                "classification_report": {"0": {"precision": 0.91, ...}, "1": {...}}
            }

    Raises:
        HTTPException: If there's a problem with the request or model training.
    """
    try:
        # --- Data Extraction and Preparation ---
        # Extract data from the request payload.
        dataset_json = request_data["dataset"]
        model_name = request_data["model_name"]
        hyperparameters = request_data["hyperparameters"]

        # Generate a unique model ID.
        model_id = str(uuid.uuid4())

        # Load the dataset from JSON.
        dataset = pd.read_json(StringIO(dataset_json), orient="split")

        # --- Check for Existing Model ---
        # Check if a model with these parameters already exists in the database.
        for file in os.listdir(MODELS_DIR):
            if file.endswith(".pkl"):  # Check for saved model files.
                id_without_extension = file[:-4]
                model_data = get_model(id_without_extension)
                if model_data is not None:
                    dataset_stored = pd.read_json(
                        StringIO(model_data[0]), orient="split"
                    )
                    if (
                        model_data[1] == model_name
                        and model_data[2] == str(hyperparameters)
                        and list(dataset.columns) == list(dataset_stored.columns)
                    ):
                        # Return existing model details if a match is found.
                        return {
                            "model_id": id_without_extension,
                            "message": "Model with these parameters already exists!",
                            "confusion_matrix": eval(model_data[3]),
                            "classification_report": eval(model_data[4]),
                            "accuracy": model_data[5],
                        }

        # --- Data Splitting ---
        # Split the dataset into features (X) and target (y).
        X = dataset.drop("target", axis=1)
        y = dataset["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Model Initialization ---
        # Initialize the model based on the provided model name.
        if model_name == "kNN - k-Nearest Neighbors":
            k = hyperparameters.get("k", 3)
            weights = hyperparameters.get("weights", "uniform")
            model = KNeighborsClassifier(n_neighbors=k, weights=weights)
        elif model_name == "SVM - Support Vector Machine":
            C = hyperparameters.get("C", 1.0)
            kernel = hyperparameters.get("kernel", "rbf")
            model = SVC(C=C, kernel=kernel)
        elif model_name == "DT - Decision Tree":
            max_depth = hyperparameters.get("max_depth", None)
            criterion = hyperparameters.get("criterion", "gini")
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        else:
            # Raise an error for an unknown model name.
            raise ValueError(f"Unknown model name: {model_name}")

        # --- Model Training and Evaluation ---
        # Train the model.
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate performance metrics.
        accuracy = accuracy_score(y_test, y_pred)
        # Convert confusion matrix to list for JSON serialization.
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        # Convert classification report to dict for JSON serialization, handling zero division.
        class_report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )

        # --- Model Persistence and Storage ---
        # Save the trained model to a file.
        model_filename = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        with open(model_filename, "wb") as model_file:
            pickle.dump(model, model_file)

        # Save model metadata to the database.
        create_model(
            model_id,
            dataset_json,
            model_name,
            hyperparameters,
            accuracy,
            conf_matrix,
            class_report,
        )

        # --- Response ---
        # Return the model ID and performance metrics.
        return {
            "model_id": model_id,
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(request_data: dict):
    """
    Endpoint to make predictions using a trained model.

    Args:
        request_data (dict): A dictionary containing the model ID and input data.
                            Example:
                            {
                                "model_id": "some-uuid",
                                "input_data": [1.2, 3.4, 5.6, 7.8]
                            }

    Returns:
        dict: A dictionary containing the prediction result.
            Example:
            {
                "prediction": 1
            }

    Raises:
        HTTPException: If there's a problem with the request or prediction.
    """
    try:
        # --- Data Extraction and Model Loading ---
        # Extract the model ID and input data from the request payload.
        model_id = request_data["model_id"]
        input_data = request_data["input_data"]

        # Load the trained model from a file.
        model_filename = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if not os.path.exists(model_filename):
            raise ValueError(f"Model ID {model_id} not found.")
        with open(model_filename, "rb") as model_file:
            model = pickle.load(model_file)

        # --- Input Data Validation ---
        # Ensure input_data is in the correct format (e.g., 2D array).
        input_data = [input_data]

        # --- Prediction ---
        # Make the prediction.
        prediction = model.predict(input_data)

        # --- Prediction Persistence ---
        # Save the prediction to the database.
        create_prediction(model_id, input_data, prediction[0])

        # --- Response ---
        # Return the prediction result.
        return {"prediction": int(prediction[0])}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
