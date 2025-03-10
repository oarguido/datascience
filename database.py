import os
import sqlite3

# --- Configuration and Setup ---

# Define the folder where the database file will be stored.
DATABASE_FOLDER = os.getenv("DATABASE_FOLDER", "db")

# Define the full path to the database file.
DATABASE = os.path.join(DATABASE_FOLDER, "models.db")


# --- Database Initialization ---


def create_table():
    """
    Creates the necessary tables in the database if they do not already exist.

    This function creates two tables:
    - models: Stores metadata about the trained machine learning models.
    - predictions: Stores the predictions made by the trained models.
    """
    # Create the folder to storage the database (if it doesn't exist)
    if not os.path.exists(DATABASE_FOLDER):
        os.makedirs(DATABASE_FOLDER)

    # Establish a connection to the SQLite database using a context manager.
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()

        # Create the 'models' table if it doesn't exist.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                dataset TEXT,
                model_name TEXT,
                hyperparameters TEXT,
                confusion_matrix TEXT,
                classification_report TEXT,
                accuracy REAL
            )
            """
        )

        # Create the 'predictions' table if it doesn't exist.
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                input_data TEXT,
                prediction REAL,
                FOREIGN KEY (model_id) REFERENCES models(id)
            )
            """
        )


# --- Model Operations ---


def create_model(
    model_id,
    dataset,
    model_name,
    hyperparameters,
    accuracy,
    confusion_matrix,
    classification_report,
):
    """
    Inserts a new trained model's metadata into the 'models' table.

    Args:
        model_id (str): The unique identifier for the model.
        dataset (str): The JSON representation of the dataset used for training.
        model_name (str): The name of the model (e.g., "kNN - k-Nearest Neighbors").
        hyperparameters (dict): The hyperparameters used during training.
        accuracy (float): The accuracy of the trained model.
        confusion_matrix (list): The confusion matrix of the trained model.
        classification_report (dict): The classification report of the trained model.
    """
    # Establish a connection to the SQLite database using a context manager.
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()

        # Insert the model's metadata into the 'models' table.
        cursor.execute(
            """
            INSERT INTO models (id, dataset, model_name, hyperparameters, confusion_matrix, classification_report, accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                dataset,
                model_name,
                str(hyperparameters),
                str(confusion_matrix),
                str(classification_report),
                accuracy,
            ),
        )


def get_model(model_id):
    """
    Retrieves a trained model's metadata from the 'models' table based on its ID.

    Args:
        model_id (str): The unique identifier of the model to retrieve.

    Returns:
        tuple: A tuple containing the model's dataset, model_name, hyperparameters,
            confusion_matrix, classification_report, and accuracy, or None if not found.
    """
    # Establish a connection to the SQLite database using a context manager.
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()

        # Retrieve the model's metadata based on the model ID.
        cursor.execute(
            "SELECT dataset, model_name, hyperparameters, confusion_matrix, classification_report, accuracy FROM models WHERE id = ?",
            (model_id,),
        )
        result = cursor.fetchone()
    return result


# --- Prediction Operations ---


def create_prediction(model_id, input_data, prediction):
    """
    Inserts a new prediction record into the 'predictions' table.

    Args:
        model_id (str): The ID of the model that made the prediction.
        input_data (list): The input data used for the prediction.
        prediction (float): The prediction result.
    """
    # Establish a connection to the SQLite database using a context manager.
    with sqlite3.connect(DATABASE) as connection:
        cursor = connection.cursor()

        # Insert the prediction record into the 'predictions' table.
        cursor.execute(
            """
            INSERT INTO predictions (model_id, input_data, prediction)
            VALUES (?, ?, ?)
            """,
            (model_id, str(input_data), prediction),
        )
