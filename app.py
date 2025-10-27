from flask import Flask, render_template, redirect, url_for, request, jsonify
import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import logging

app = Flask(__name__)

# Get Streamlit URL from env var (default: http://localhost:8501)
STREAMLIT_URL = os.getenv('STREAMLIT_URL', 'http://localhost:8501')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API key (assuming GEMINI_API_KEY is in env)
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("Missing GEMINI_API_KEY in environment variables")

# Initialize LLM
langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

# Chat history
student_assistant_history = InMemoryChatMessageHistory()

def get_session_history(assistant_type: str):
    MAX_MESSAGES = 5
    histories = {
        'student': student_assistant_history
    }
    history = histories.get(assistant_type, student_assistant_history)
    if len(history.messages) > MAX_MESSAGES:
        history.messages = history.messages[-MAX_MESSAGES:]
    return history

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    # Redirect to Streamlit dashboard
    return redirect(STREAMLIT_URL)

@app.route('/students')
def students():
    # Placeholder for students route - redirect or render as needed
    return render_template('students.html')

@app.route('/simulation')
def simulation():
    # Placeholder for simulation route
    return render_template('simulation.html')

@app.route('/student_assistant')
def student_assistant():
    lang = request.args.get('lang', 'en')
    messages = [
        {"role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant", "content": msg.content}
        for msg in get_session_history('student').messages
    ]
    return render_template('student_assistant.html', lang=lang, messages=messages)

@app.route('/student_assistant', methods=['POST'])
def student_assistant_post():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['userInput', 'language']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({"error": f"Missing or empty field: {field}"}), 400

        query = data['userInput'].strip()
        language = data['language'].strip().lower()

        valid_languages = ['en', 'hi', 'kn']
        if language not in valid_languages:
            logger.warning(f"Invalid language '{language}', defaulting to 'en'")
            language = 'en'

        language_names = {'en': 'English', 'hi': 'Hindi', 'kn': 'Kannada'}
        language_name = language_names[language]

        prompt_template = PromptTemplate(
            input_variables=["query", "language_name"],
            template=""" 
            You are a friendly AI student guidance assistant for Indian students. Provide a clear and concise answer to the following question related to scholarships, courses, or educational opportunities. The answer must be in {language_name} and tailored to the Indian context (e.g., referencing Indian scholarships, courses, or local educational institutions). Limit the response to 3-5 sentences for brevity. Use simple language suitable for students and avoid complex jargon. If the question is too vague or unrelated to student guidance, return a polite message indicating the need for a more specific question.

            Question: {query}

            Instructions:
            - Answer in {language_name}, using simple and clear language.
            - Focus on practical advice or information relevant to scholarships, courses, or educational opportunities in India.
            - If unsure, suggest consulting a local school counselor or education department.
            - Do not include markdown, code fences, or additional text—only the plain text response.
            - Do not use any special symbols like *
            - If user is greeting, then you also greet

            Example (for English):
            For scholarships, check the National Scholarship Portal or state education department. Apply online with your marksheets and income certificate. Many schemes like PM-USP for girls or Post-Matric for SC/ST students offer full support. Visit your local block education office for application help.
            """
        )

        prompt = prompt_template.format(
            query=query,
            language_name=language_name
        )

        try:
            logger.debug(f"Sending prompt to Gemini: {prompt[:200]}...")
            response = langchain_llm.invoke(prompt)
            logger.debug(f"Gemini response: {response.content}")

            answer = response.content.strip()
            if not answer:
                logger.warning("Empty response from Gemini")
                answer = "No answer found. Please ask a more specific student guidance question."

            session_history = get_session_history('student')
            session_history.add_user_message(query)
            session_history.add_ai_message(answer)

            return jsonify({"response": answer}), 200

        except Exception as e:
            logger.error(f"Gemini query error: {str(e)}")
            return jsonify({"error": "Error processing query. Please try again later."}), 503

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/about')
def about():
    # Placeholder for about route
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)