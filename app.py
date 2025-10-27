from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    # Placeholder for dashboard route - implement with Streamlit or additional Flask views
    return render_template('dashboard.html')  # Assume a dashboard.html exists or redirect

@app.route('/students')
def students():
    # Placeholder for students route
    return render_template('students.html')

@app.route('/simulation')
def simulation():
    # Placeholder for simulation route
    return render_template('simulation.html')

@app.route('/chatbot')
def chatbot():
    # Placeholder for chatbot route
    return render_template('chatbot.html')

@app.route('/about')
def about():
    # Placeholder for about route
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)