from flask import Flask, render_template, redirect, url_for, request, jsonify
import os
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from mem0 import MemoryClient
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any, TypedDict, Optional
import logging
from dotenv import load_dotenv
import json
import re

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
serper_api_key = os.getenv('SERPER_API_KEY')

if not all([gemini_api_key, mem0_api_key]):
    raise ValueError("Missing one or more API keys: GEMINI_API_KEY, MEM0_API_KEY")

# Initialize LLM
langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=gemini_api_key
)

serper_tool = SerperDevTool(
    api_key=serper_api_key,
    n_results=50
)

llm = LLM(
    model="gemini/gemini-2.5-flash",
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

# ... (existing imports and setup remain the same)

class Scholarship(BaseModel):
    name: str
    category: str  # e.g., Merit-based, Need-based, SC/ST
    description: str
    link: str

class ScholarshipSearchTool(BaseTool):
    name: str = Field(default="ScholarshipSearchTool", description="Searches for scholarships using web search.")
    description: str = Field(default="Fetches scholarship details like name, category, description, and application link using SerperDevTool.")

    def _run(self, query: str) -> List[Dict[str, str]]:
        try:
            # Broaden search: Remove strict site limits, add more general terms for better results
            search_query = f"scholarships for {query} in India 2025 undergraduate postgraduate merit need-based SC ST OBC"
            search_results = serper_tool._run(query=search_query, n_results=20)  # Increase to 20 for more hits
            
            scholarships = []
            for result in search_results.get('organic', [])[:15]:  # Top 15 relevant results
                title = result.get('title', 'Unknown Scholarship')
                snippet = result.get('snippet', 'No detailed description available.')
                link = result.get('link', '#')
                
                # Ensure snippet has content; fallback if empty
                if not snippet or len(snippet.strip()) < 10:
                    snippet = f"Scholarship opportunity related to {query}. Visit the link for eligibility and application details."
                
                # Skip irrelevant results
                if 'scholarship' not in title.lower() and 'scholarship' not in snippet.lower():
                    continue
                
                scholarships.append({
                    'name': title,
                    'category': self._extract_category(snippet),
                    'description': snippet[:250] + '...' if len(snippet) > 250 else snippet,
                    'link': link
                })
            
            # If no results, add a fallback general scholarship with description
            if not scholarships:
                scholarships = [{
                    'name': 'National Scholarship Portal - General Search',
                    'category': 'General',
                    'description': 'The official Government of India portal aggregates all central and state scholarships. Users can search by category, education level, and eligibility criteria such as income and caste. Register to apply for multiple schemes in one place.',
                    'link': 'https://scholarships.gov.in/'
                }]
            
            return scholarships
        except Exception as e:
            logger.error(f"Error searching scholarships: {str(e)}")
            # Fallback list of common scholarships with detailed descriptions
            return [{
                'name': 'Post Matric Scholarship for SC/ST',
                'category': 'Category-based',
                'description': 'This scheme provides financial assistance to SC/ST students pursuing post-matriculation courses. It covers tuition fees, examination fees, and maintenance allowance based on group classification. Eligible for students from families with income below specified limits.',
                'link': 'https://scholarships.gov.in/'
            }, {
                'name': 'Central Sector Scheme of Scholarship for College and University Students',
                'category': 'Merit-based',
                'description': 'For meritorious students from low-income families (income < 4.5 lakh/year) who scored 80th percentile in Class XII. Provides Rs. 10,000-20,000 per annum for UG/PG courses. Apply through National Scholarship Portal.',
                'link': 'https://scholarships.gov.in/'
            }, {
                'name': 'INSPIRE Scholarship',
                'category': 'Merit-based',
                'description': 'Department of Science & Technology scheme for top 1% students in sciences at Class XII. Offers Rs. 80,000 per year for higher education in natural/basic sciences. Encourages research careers.',
                'link': 'https://www.online-inspire.gov.in/'
            }]

    def _extract_category(self, snippet: str) -> str:
        snippet_lower = snippet.lower()
        if any(word in snippet_lower for word in ['merit', 'academic']): return 'Merit-based'
        if any(word in snippet_lower for word in ['need', 'income', 'financial']): return 'Need-based'
        if any(word in snippet_lower for word in ['sc', 'st', 'obc', 'minority', 'ews']): return 'Category-based'
        return 'General'

class ScholarshipFilterTool(BaseTool):
    name: str = Field(default="ScholarshipFilterTool", description="Filters scholarships based on user criteria with priority.")
    description: str = Field(default="Filters scholarships by prioritizing education level, income, caste, gender, location.")

    def _run(self, scholarships: List[Dict[str, Any]], user_data: dict) -> List[Dict[str, Any]]:
        try:
            filtered_scholarships = []
            education_level = user_data.get('education_level', '').lower()
            income = user_data.get('family_income', 0)
            caste = user_data.get('caste', '').lower()
            gender = user_data.get('gender', '').lower()
            location = user_data.get('location', '').lower()

            for scholarship in scholarships:
                name_lower = scholarship.get('name', '').lower()
                category_lower = scholarship.get('category', '').lower()
                desc_lower = scholarship.get('description', '').lower()

                # Less strict matching: education level > income > caste/gender > location
                education_match = education_level in name_lower or education_level in desc_lower or any(lev in desc_lower for lev in ['ug', 'pg', 'undergraduate', 'postgraduate'])
                income_match = True  # Less strict: always match for income, or adjust threshold dynamically
                if 'need' in category_lower and income > 800000:  # Higher threshold for need-based
                    income_match = False
                category_match = caste in desc_lower or gender in desc_lower or 'general' in category_lower
                location_match = location in desc_lower or 'india' in desc_lower.lower() or not location  # Always match if no location

                score = sum([bool(education_match), bool(income_match), bool(category_match), bool(location_match)])
                
                # Less strict: >=1 match, or return all if none
                if score >= 1 or not filtered_scholarships:
                    # Ensure description is always present and detailed
                    if len(scholarship.get('description', '')) < 50:
                        scholarship['description'] = f"Eligibility: Matches your {education_level} level and {caste} category. Benefits include fee reimbursement and stipends. Apply via official portal for full details."
                    filtered_scholarships.append(scholarship)

            # If still empty, return all input scholarships
            if not filtered_scholarships:
                filtered_scholarships = scholarships[:10]

            # Sort by name for consistency
            return sorted(filtered_scholarships, key=lambda x: x.get('name', ''), reverse=False)[:10]  # Top 10
        except Exception as e:
            logger.error(f"Error filtering scholarships: {str(e)}")
            return scholarships[:10]  # Fallback to first 10

# CrewAI Setup for Scholarships (adapt from schemes)
def create_scholarship_crew(user_data: dict):
    # Search Agent
    search_agent = Agent(
        role='Scholarship Search Specialist',
        goal='Search for relevant scholarships based on user criteria like education level, income, caste, and location.',
        backstory='You are an expert in Indian educational scholarships, using reliable sources to find opportunities.',
        tools=[ScholarshipSearchTool()],
        llm=llm,
        verbose=True
    )

    # Filter Agent
    filter_agent = Agent(
        role='Scholarship Filter Expert',
        goal='Filter and prioritize scholarships that best match the user\'s profile.',
        backstory='You analyze search results and apply filters for eligibility, relevance, and priority.',
        tools=[ScholarshipFilterTool()],
        llm=llm,
        verbose=True
    )

    # Tasks with expected_output
    search_task = Task(
        description=f'Search for scholarships matching: {user_data}. Return a list of potential scholarships.',
        expected_output="A list of 10-15 scholarships with name, category, description, and link.",
        agent=search_agent
    )

    filter_task = Task(
        description=f'Filter the searched scholarships using user data: {user_data}. Prioritize by education level, income, caste, gender, location. Return top 5-10 matches.',
        expected_output="A filtered list of 5-10 most relevant scholarships based on user criteria, including name, category, description, and link.",
        agent=filter_agent
    )

    # Crew with verbose=True
    crew = Crew(
        agents=[search_agent, filter_agent],
        tasks=[search_task, filter_task],
        verbose=True
    )

    return crew

@app.route('/scholarship_finder', methods=['GET', 'POST'])
def scholarship_finder():
    if request.method == 'POST':
        try:
            # Get form data
            user_data = {
                'name': request.form.get('name', '').strip(),
                'age': int(request.form.get('age', 0)),
                'caste': request.form.get('caste', '').strip(),
                'location': request.form.get('location', '').strip(),
                'education_level': request.form.get('education_level', '').strip(),
                'family_income': float(request.form.get('family_income', 0)),
                'gender': request.form.get('gender', '').strip()
            }

            if not all([user_data['name'], user_data['age'], user_data['location'], user_data['education_level'], user_data['family_income']]):
                return render_template('scholarship_finder.html', error_message="Please fill in all required fields.", form_submitted=True)

            # Run CrewAI
            crew = create_scholarship_crew(user_data)
            result = crew.kickoff()

            # Parse result more robustly
            scholarships = []
            import json
            import re
            
            raw_result_str = str(result).strip() # Ensure we're working with a string
            logger.info(f"Raw CrewAI result received: {raw_result_str[:200]}") # Debug log

            # --- START FIX: RE-ORDERED PARSING LOGIC ---

            # FIX 1: Prioritize JSON parsing as it's the expected output
            json_match = re.search(r'\[.*\]', raw_result_str, re.DOTALL)
            
            if json_match:
                logger.info("Attempting JSON parsing...")
                try:
                    raw_scholarships = json.loads(json_match.group())
                    if isinstance(raw_scholarships, list):
                        scholarships = [Scholarship(**sch).dict() for sch in raw_scholarships if isinstance(sch, dict)]
                        logger.info(f"Successfully parsed {len(scholarships)} scholarships from JSON.")
                except json.JSONDecodeError:
                    logger.warning("CrewAI output looked like JSON but failed to parse. Will try regex.")
                    scholarships = [] # Reset on failure

            # FIX 2: Only run markdown regex parsing if JSON parsing failed or wasn't applicable
            if not scholarships and not raw_result_str.startswith('['):
                logger.info("JSON parsing failed or skipped. Attempting Markdown regex parsing...")
                # Extract numbered items (e.g., "1. **Name** * Category * Description * Link")
                items = re.split(r'\n(?=\d+\.\s+)', raw_result_str)  # Split on new numbered items
                for item in items:
                    if not item.strip():
                        continue
                    
                    # Extract name (after number, before first * )
                    name_match = re.search(r'\d+\.\s+\*\*(.+?)\*\*', item)
                    
                    # FIX 3: If name isn't found, it's not a valid item. Skip it.
                    if not name_match:
                        continue 
                        
                    name = name_match.group(1).strip()
                    
                    # Extract category
                    cat_match = re.search(r'\*\*\s*Category:\s*(.+?)(?=\n|\*\*)', item, re.DOTALL)
                    category = cat_match.group(1).strip() if cat_match else 'General'
                    
                    # Extract description (between category and link)
                    desc_match = re.search(r'(?<=Description:)(.+?)(?=Link:|\n\d+)', item, re.DOTALL)
                    description = desc_match.group(1).strip() if desc_match else 'No description available.'
                    
                    # Extract link
                    link_match = re.search(r'Link:\s*`(.+?)`', item)
                    link = link_match.group(1).strip() if link_match else 'https://scholarships.gov.in/'
                    
                    scholarships.append({
                        'name': name,
                        'category': category,
                        'description': description,
                        'link': link
                    })
                if scholarships:
                    logger.info(f"Successfully parsed {len(scholarships)} scholarships from Markdown regex.")

            # FIX 4: Handle if crew.kickoff() returned a direct list (not a string)
            elif not scholarships and isinstance(result, list):
                 logger.info("CrewAI result was a direct list object.")
                 scholarships = [Scholarship(**sch).dict() for sch in result if isinstance(sch, dict)]

            # --- END FIX ---

            # Limit to top 8-10
            scholarships = scholarships[:10]
                
            # Ensure descriptions are detailed and present
            for sch in scholarships:
                if len(sch.get('description', '')) < 50:
                    sch['description'] = f"Detailed eligibility: Suitable for {user_data['education_level']} students with family income under Rs. {user_data['family_income']:,}. Covers tuition, books, and stipends. Check link for application deadlines and documents required."

            # Ensure at least some results (fallback)
            if not scholarships:
                logger.warning("No scholarships parsed, using hardcoded fallback list.") # Log this
                scholarships = [{
                    'name': 'National Scholarship Portal',
                    'category': 'General',
                    'description': 'The official Government of India portal aggregates all central and state scholarships. Users can search by category, education level, and eligibility criteria such as income and caste. Register to apply for multiple schemes in one place, with step-by-step guidance and document checklists.',
                    'link': 'https://scholarships.gov.in/'
                }, {
                    'name': 'Post-Matric Scholarship Scheme',
                    'category': 'Category-based',
                    'description': 'Financial aid for SC/ST/OBC/EWS students in post-matric courses like UG/PG/Diploma. Reimbursement of tuition fees up to Rs. 20,000, plus maintenance allowance based on hostel/day scholar status. Income limit: Rs. 2.5 lakh for most categories. Apply annually via NSP.',
                    'link': 'https://scholarships.gov.in/'
                }]

            # Debug log
            logger.info(f"Final parsed {len(scholarships)} scholarships: {[sch['name'] for sch in scholarships]}")

            return render_template('scholarship_finder.html', scholarships=scholarships, form_submitted=True)

        except Exception as e:
            logger.error(f"Error in scholarship finder: {str(e)}")
            # Fallback scholarships on error with descriptions
            fallback_scholarships = [{
                'name': 'Central Sector Scholarship',
                'category': 'Merit-based',
                'description': 'Awarded to top-performing students from families with income < Rs. 4.5 lakh. Provides Rs. 12,000 (UG) or Rs. 20,000 (PG) per year for tuition and non-refundable fees. Based on Class XII marks; no separate application—auto-selected via NSP.',
                'link': 'https://scholarships.gov.in/'
            }, {
                'name': 'Ishān Udāy Scholarship',
                'category': 'Region-based',
                'description': 'For students from North-Eastern states pursuing professional courses. Covers full tuition and incidental fees up to Rs. 7,800/month. Eligibility: 60% marks in qualifying exam; preference for technical fields like engineering/medicine.',
                'link': 'https://scholarships.gov.in/'
            }]
            return render_template('scholarship_finder.html', scholarships=fallback_scholarships, form_submitted=True, error_message="Using fallback results due to temporary issue.")

    return render_template('scholarship_finder.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)