# AIDI Machine Learning Project: Model Training and Prediction

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

This project demonstrates a simple machine learning application that allows you to train various models (k-Nearest Neighbors, Support Vector Machine, and Decision Tree) on datasets (Iris or custom CSV) and make predictions using the trained models. The application consists of a Streamlit frontend, a FastAPI backend, and an SQLite database. The application is fully containerized using Docker.

## Project Components

*   **`app.py` (Frontend - Streamlit):** Provides the user interface for data selection, model selection, hyperparameter tuning, model training, and displaying results.
*   **`main.py` (Backend - FastAPI):** Defines the API endpoints for model training and prediction. It handles the logic for interacting with the database, storing models, and training machine learning models.
*   **`database.py` (Database):** Contains the code for database setup, model definition, and database interaction functions. It provides an abstraction layer for working with the database, making it easier to manage and maintain.
*   **`models/` (Directory):** Stores the trained machine learning models as pickle (`.pkl`) files.
*   **`db/` (Directory):** Contains the SQLite database file (`models.db`) used to store model information and predictions.
*   **`.env` (File):** Stores environment variables (e.g., ports, directory paths) that configure both the frontend and backend.
*   **`requirements.txt` (File):** Lists all Python dependencies required to run both the FastAPI backend and the Streamlit frontend.
*   **`docker-compose.yml` (File):** Defines the Docker Compose configuration for building and running the application containers.
*   **`Dockerfile.fastapi`:** The Dockerfile for building the FastAPI backend image.
*   **`Dockerfile.streamlit`:** The Dockerfile for building the Streamlit frontend image.

## Features

*   **Dataset Selection:** Choose between the built-in Iris dataset or upload your own CSV file (with a 'target' column).
*   **Model Selection:** Train k-Nearest Neighbors (kNN), Support Vector Machine (SVM), or Decision Tree (DT) models.
*   **Hyperparameter Tuning:** Adjust model hyperparameters through sliders and select boxes.
*   **Model Training:** Train selected models with the chosen dataset and hyperparameters.
*   **Metric Display:** View model metrics (F1-Score, confusion matrix, classification report).
*   **Prediction:** Make predictions on new data using a trained model.
*   **Model Cleanup:** Clear trained models (pickle files).
*   **Containerized:** The application is fully containerized using Docker containers.
*   **FastAPI:** The backend is built using FastAPI.
*   **Streamlit:** The frontend is built using Streamlit.
*   **Database:** SQLite is used as the database.

## Prerequisites

*   **Python 3.11+**
*   **Docker and Docker Compose** (for containerized deployment)
*   **Git** (for version control and interacting with GitHub)

## Installation and Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/oarguido/aidi-project.git
    ```
    Remember to replace it with your repo path.

2.  **Navigate to the project directory:**

    ```bash
    cd aidi-project
    ```

3.  **Create a `.env` file:**

    *   Create a file named `.env` in the project's root directory.
    *   Add the following lines, adjusting the values if necessary:

        ```
        FASTAPI_PORT=8000
        STREAMLIT_PORT=8501
        MODELS_DIR=models
        DATABASE_FOLDER=db
        ```

        * **`FASTAPI_PORT`**: The port where the FastAPI backend will be accessible from outside the container. (default: 8000). If this port is already in use, you may need to change it.
        * **`STREAMLIT_PORT`**: The port where the Streamlit frontend will be accessible from outside the container. (default: 8501). If this port is already in use, you may need to change it.
        * **`MODELS_DIR`**: The path where the model files (`.pkl`) are going to be stored. (default: `models`).
        * **`DATABASE_FOLDER`**: The path where the database file (`.db`) is going to be stored. (default: `db`).

4.  **Create `models` and `db` folders**: Create both folders in the root directory of your project.

5. **Verify the files:** Verify that the following files are present in the root folder of your project:

    * `app.py`
    * `main.py`
    * `database.py`
    * `requirements.txt`
    * `.env`
    * `docker-compose.yml`
    * `Dockerfile.fastapi`
    * `Dockerfile.streamlit`
    * `models` (folder)
    * `db` (folder)

## Running the Application (Docker Compose)

1.  **Build and run the Docker containers:**

    ```bash
    docker compose up --build -d
    ```

    This will:

    *   Build the Docker images for the `fastapi` (backend) and `streamlit` (frontend) services, if they don't exist or if there are changes in the code.
    *   Start the containers for both services in detached mode (`-d`).
    *   The first time it will take some time to build the images.

2.  **Access the application:**

    *   **Streamlit Frontend:** Open your web browser and go to `http://localhost:8501` (or the port you defined in the `STREAMLIT_PORT` variable).
    *   **FastAPI Backend:** The FastAPI documentation will be available at `http://localhost:8000/docs` (or the port you defined in the `FASTAPI_PORT` variable).

3. **Stop the application:**
    ```bash
    docker compose down
    ```

## Using the Application

1.  **Choose a Dataset:** In the Streamlit app, select "Iris Dataset" or "Upload a CSV file." If you upload a CSV, make sure it has a `target` column.
2.  **Choose a Model:** Select the machine learning model you want to train.
3.  **Set Hyperparameters:** Adjust the hyperparameters for the selected model using the provided controls.
4.  **Train the Model:** Click the "Train Model" button. You'll see the model's metrics (F1-Score, confusion matrix and classification report) once training is complete.
5.  **Make a Prediction:** Enter input values for each feature (sepal length, sepal width, petal length and petal width) to make a prediction with the trained model.
6. **Clean up:** On the sidebar you will find a "Clear Models" button. You can use it to remove the pickle files that are stored on the `models` folder.

## Cleanup

* **Clear Models:** The "Clear Models" button in the sidebar will remove all the trained models (pickle files) from the `models` folder.
* **Stop the Application:** Use the `docker compose down` command to stop the application and remove the containers.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT License

## Contact

André Lima / University of Aveiro - https://orcid.org/my-orcid?orcid=0000-0001-5505-1233