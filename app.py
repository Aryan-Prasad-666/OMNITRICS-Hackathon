from flask import Flask, render_template, redirect, url_for, request, jsonify
import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from mem0 import MemoryClient
import logging
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Get Streamlit URL from env var (default: http://localhost:8501)
STREAMLIT_URL = os.getenv('STREAMLIT_URL', 'http://localhost:8501')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load API keys
gemini_api_key = os.getenv('GEMINI_API_KEY_S')
mem0_api_key = os.getenv('MEM0_API_KEY')

if not all([gemini_api_key, mem0_api_key]):
    raise ValueError("Missing one or more API keys: GEMINI_API_KEY, MEM0_API_KEY")

# Initialize LLM
langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

# Initialize Mem0
mem0_client = MemoryClient(api_key=mem0_api_key)

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
            You are a helpful Assistant for educators in Indian higher education institutions. Provide clear, actionable advice on student engagement, risk prediction, and interventions based on academic data (e.g., attendance, marks, socio-economic factors). Tailor responses to the Indian college context, referencing tools like the National Scholarship Portal or counseling best practices. Limit to 3-5 sentences. Use simple language for faculty/admins. If unrelated, politely redirect to prediction insights.

            Query: {query}

            Instructions:
            - Respond in {language_name}.
            - Suggest data-driven interventions (e.g., 'For a high-risk student with low attendance, schedule a mentoring session and flag for financial aid check').
            - Reference ML outputs like risk scores or feature impacts if relevant.
            - End with next steps, like 'Check the Student Detail View for more insights'.
            - No markdown, code, or symbols—plain text only.
            - Greet if starting a conversation.

            Example (English): Based on the 65% disengagement risk from low internal marks and attendance, recommend a personalized study plan and connect them to peer mentoring. Use the Simulation Panel to test intervention outcomes. Monitor progress via weekly check-ins to prevent dropout.
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

            # Store the conversation in Mem0
            mem0_client.add(
                messages=[
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": answer}
                ],
                user_id="student_assistant_user",
                output_format="v1.1"
            )

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