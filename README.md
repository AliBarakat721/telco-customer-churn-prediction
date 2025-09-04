# Telco Customer Churn Prediction

This project is a comprehensive analysis and predictive modeling solution for customer churn in a telecommunications company. The primary goal is to identify customers who are likely to cancel their subscriptions, enabling the business to proactively engage them with retention strategies.

---

## 🚀 Project Highlights

- **End-to-End Workflow:** Covers the entire data science lifecycle, from data cleaning and EDA to model training and evaluation.
- **Predictive Modeling:** Implements and compares `RandomForestClassifier` and `XGBoost` to find the best performer.
- **Interactive Web API:** Includes a web application built with **FastAPI** that serves the trained model, allowing for real-time predictions.
- **Ready for Deployment:** The project includes saved (pickled) models, encoders, and scalers, making it ready for production.

---

## 🛠️ Technologies & Libraries Used

- **Python 3.x**
- **Pandas & NumPy:** For data manipulation.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** For machine learning.
- **FastAPI:** For building the web API.
- **Uvicorn:** As the ASGI server to run the API.
- **Jupyter Notebook:** For interactive analysis.

---

## 📂 Project Structure

.
├── .gitignore
├── Customer Churn Prediction .ipynb
├── README.md
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── best_model.pkl
├── encoder.pkl
├── requirements.txt
├── scaler.pkl
└── fastapi_app.py  <-- The Web Application API
Plain Text

---

## 🏁 How to Get Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

Make sure you have Python 3 and Git installed on your system.

### 2. Clone the Repository

```bash
git clone https://github.com/AliBarakat721/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
3. Set Up a Virtual Environment
Bash
# Create the virtual environment
python -m venv venv

# Activate it (on Windows )
venv\Scripts\activate
4. Install Dependencies
Bash
pip install -r requirements.txt
5. Run the Web Application (API)
To start the interactive web application, run the following command in your terminal:
Bash
uvicorn fastapi_app:app --reload
uvicorn: The server that runs the application.
fastapi_app: The name of your Python file (fastapi_app.py).
app: The name of the FastAPI object you created inside the file (app = FastAPI()).
--reload: Automatically restarts the server when you make changes to the code.
6. Access the Application
Once the server is running, open your web browser and go to:
http://127.0.0.1:8000
You will see the web interface where you can input customer data and get a churn prediction.
📈 Key Results
The final model (RandomForestClassifier ) achieved the following performance on the unseen test set:
Accuracy: ~78%
Recall (for "Churn=Yes"): ~67%
ROC-AUC Score: ~74%
The model demonstrates a good ability to identify two-thirds of the customers who are at risk of churning, providing a valuable tool for the business.
