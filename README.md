# Streamlit AI Fitness Chatbot & Progress Tracker

This project is a specialized Streamlit-based AI Chatbot designed for the fitness and health industry. It functions as a virtual gym instructor that provides personalized workout guidance,       tracks exercise performance, and offers data-driven progress insights through a conversational interface.

# Key Features
  📍  Personalized Routine Generation: Recommends workouts tailored to specific user goals such as strength, weight loss, endurance, or flexibility.
  📍  Conversational Logging: Allows users to log completed workouts and sets using simple natural language messages.
  📍  Automated Progress Analytics: Calculates total workouts completed, total hours exercised, and the percentage of the current goal achieved.
  📍  Predictive Tracking: Estimates goal completion dates and answers projection-based questions, such as total expected workouts by a specific future date.
  📍  Instructional Guidance: Provides step-by-step exercise instructions, including tips for form, intensity, and variations.
  📍  Weekly Performance Summaries: Generates a summary every Sunday detailing workouts completed and progress toward long-term goals.

# User Experience & Interface
  The application is designed with an intuitive dual-component layout:
  📲  Fitness Profile Sidebar: A dedicated space for users to input their fitness level (Beginner, Intermediate, Advanced), workout frequency, and any physical limitations or injuries.
  📲  AI Chat Window: An interactive space where the bot provides motivational greetings, answers fitness/nutrition questions, and logs daily activity.

# Safety & Scope
  To ensure user safety, the chatbot operates under the following strictly defined guardrails:
  ✔️  No Medical Advice: The bot is explicitly programmed not to provide medical advice or replace the role of a certified professional.
  ✔️  Injury Prevention: The system suggests modifications if exercises are deemed too difficult and advises consulting a doctor for existing health concerns.
  ✔️  Targeted Assistance: Designed specifically for gym members, fitness beginners, and intermediate users.
  ✔️  Realistic Expectations: The bot does not make exaggerated claims or guarantee specific physical results.

# System Tone & Output
  The chatbot maintains a friendly, concise, and motivating tone. Every interaction is designed to include:
  📜  A motivational greeting.
  📜  Clear, step-by-step instructions.
  📜  A relevant progress summary when data is updated.


# This guide will help you get the Streamlit AI Fitness Chatbot up and running on your local machine. This application serves as a conversational gym instructor that provides personalized workout guidance, tracks exercise performance, and offers data-driven progress insights.

  ➡️ 1. Clone the Repository
      Open your terminal or command prompt and run the following commands to clone the project and enter the directory:
      ```bash
      git clone <your-repository-url>
      cd <repository-folder-name>
      ```
  
  ➡️ 2. Set Up a Virtual Environment (Recommended)
      To keep your global Python installation clean, it is best practice to use a virtual environment:
      ```bash
      # Create the environment
      python -m venv venv
# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate
      ```

  ➡️ 3. Install Dependencies
      Install the required libraries, including Streamlit and the Google Generative AI SDK, using the provided requirements file:
```bash
python -m pip install -r requirements.txt
```
      

