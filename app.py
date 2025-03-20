from flask import Flask
from config import PORT, DEBUG
from routes import main_blueprint
from langchain_utils.qa_chain import initialize_app  # initialization sets up global variables

app = Flask(__name__, template_folder="templates")
app.register_blueprint(main_blueprint)

# Initialize the LangChain vectorstore and QA chain before handling any requests
initialize_app()

if __name__ == "__main__":
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)