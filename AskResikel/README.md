# Chatbot Project

This project is a chatbot application designed to recognize and classify user inputs into predefined intents such as trash recycling inquiries and swearing detection. It leverages machine learning (Support Vector Classifier - SVC) for intent recognition and provides automated responses based on the input.

## Key Features:

1. **Intent Recognition**:
   - The chatbot recognizes user input and classifies it into specific intents such as questions about trash recycling or inappropriate language.
   - It uses **TF-IDF** and **Support Vector Machines (SVM)** to process and classify user inputs.

2. **Machine Learning Model**:
   - The chatbot is trained on a dataset (`chatbot_dataset.json`) containing various patterns and responses.
   - It predicts user intent by applying a **linear kernel SVC model** and uses **cosine similarity** for finding the most relevant patterns.

3. **Response Generation**:
   - The chatbot generates responses based on the identified intent. Responses are predefined for each intent, and the chatbot selects one randomly.
   - If the similarity score between the input and known patterns is low, the chatbot will ask for clarification.

4. **API Integration**:
   - The chatbot provides a REST API endpoint (`/chat`) for interacting with the chatbot. It can handle POST requests with user input and return JSON responses.
   - The API is built using **Flask** and **Flask-RESTful**, making it easy to integrate with other services or front-end interfaces.

5. **Evaluation and Performance**:
   - The model is evaluated with a **confusion matrix** and a **classification report** that provides insight into the model's accuracy, precision, and recall across various intents.

6. **Web Interface**:
   - A simple web interface is included for interacting with the chatbot directly from a browser.

7. **Docker Support**:
   - The application includes a Docker configuration, making it easy to containerize and deploy the chatbot in different environments.

## Use Cases:
- **Trash Recycling**: Users can ask questions about trash disposal, recycling methods, or guidelines.
- **Swearing Detection**: The chatbot can detect inappropriate language and respond accordingly to maintain a respectful conversation.

## Example Workflow:

1. **User Input**: A user asks, "What should I do with plastic bottles?"
2. **Intent Classification**: The model classifies the question as a "recycle" intent.
3. **Response Generation**: Based on the "recycle" intent, the chatbot responds: "You should recycle plastic bottles by placing them in the designated recycling bin."
4. **Clarification**: If the input is unclear or has low similarity to known patterns, the chatbot will ask: "Sorry, I didn't understand. Could you please rephrase your question?"

## Tech Stack

- **Backend**: Flask (Python web framework)

## Requirements

- Python 3.10+
- Docker (optional for containerization)

## Installation

### Backend (FastAPI)

1. Clone the repository:
```bash
   git https://github.com/BudimanZahri/Sembatbes-Ai.git
   cd Sembatbes-Ai
   ```
2. Create a virtual environment:
```bash
   python -m venv bmzenv
   source bmzenv/bin/activate  # On Windows use `bmzenv\Scripts\activate`
   ```
3. Install the dependencies:
```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI server:
```bash
   python app.py
   ```

## Usage

1.  Start the Flask backend server (see above).
2.  Access the application at `http://localhost:5000` to manage your Chatbot server.

## License

This project is licensed under the [GPL-3.0 license](LICENSE).

## Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m 'Add your feature'`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

## Roadmap

- [ ] Add user authentication and role-based access control
- [ ] Improve error handling and logging
- [ ] Add real-time peer status monitoring
- [ ] Create Docker setup for easier deployment

## Contact

For issues or feature requests, feel free to open an issue on the GitHub repository or contact the maintainer.

**Maintainer**: Budiman Zahri
**Email**: budimanzahri@outlook.com
