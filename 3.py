import streamlit as st
import joblib
import re
import string
import google.generativeai as genai

# Configure Gemini API Key
GEMINI_API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

# Load the trained model
model = joblib.load("mental_health_model.pkl")

# Function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Function to get response from Gemini AI
def get_gemini_response(user_message):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(user_message)
    return response.text.strip()

# Mental health condition explanations
condition_info = {
    "Stress": {
        "symptoms": "Irritability, fatigue, headaches, trouble sleeping, difficulty concentrating.",
        "why": "Stress happens due to work pressure, personal issues, or sudden life changes.",
        "how_to_tackle": "Try relaxation techniques like meditation, exercise, and proper sleep.",
        "steps": "1. Identify stressors. 2. Practice deep breathing. 3. Take breaks and engage in hobbies.",
        "prevention": "Maintain a balanced lifestyle and manage time effectively."
    },
    "Depression": {
        "symptoms": "Persistent sadness, loss of interest, fatigue, changes in appetite, suicidal thoughts.",
        "why": "Depression can be triggered by genetics, chemical imbalances, trauma, or prolonged stress.",
        "how_to_tackle": "Cognitive Behavioral Therapy (CBT), antidepressants, regular physical activity.",
        "steps": "1. Talk to someone you trust. 2. Engage in daily activities. 3. Maintain a routine and healthy diet.",
        "prevention": "Build strong social connections and practice self-care."
    },
    "Bipolar disorder": {
        "symptoms": "Extreme mood swings, manic episodes, depressive episodes, impulsivity.",
        "why": "Caused by genetic factors, neurotransmitter imbalances, or environmental triggers.",
        "how_to_tackle": "Mood stabilizers, therapy, and lifestyle modifications can help.",
        "steps": "1. Monitor mood patterns. 2. Follow a structured routine. 3. Seek medical support.",
        "prevention": "Maintain a healthy routine and avoid triggers."
    },
    "Personality disorder": {
        "symptoms": "Unstable relationships, intense emotions, impulsivity, difficulty trusting others.",
        "why": "Develops due to genetic predisposition, childhood trauma, or environmental influences.",
        "how_to_tackle": "Psychotherapy, mindfulness, and structured daily routines aid in managing symptoms.",
        "steps": "1. Identify triggers. 2. Develop coping strategies. 3. Work with a therapist regularly.",
        "prevention": "Early intervention, therapy, and self-awareness can help reduce severity."
    },
    "Anxiety": {
        "symptoms": "Excessive worry, restlessness, rapid heartbeat, sweating, difficulty sleeping.",
        "why": "Arises from stress, trauma, overthinking, or neurochemical imbalances.",
        "how_to_tackle": "Practice mindfulness, deep breathing, and exposure therapy.",
        "steps": "1. Challenge negative thoughts. 2. Engage in relaxation exercises. 3. Seek professional help.",
        "prevention": "Limit caffeine, practice self-care, and maintain a balanced lifestyle."
    }
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_prediction_done" not in st.session_state:
    st.session_state.first_prediction_done = False  # Flag to track first prediction

# Streamlit UI
st.title("🧠 AI-Powered Mental Health Chatbot")
st.write("Chat with me about your thoughts and emotions. I will analyze your responses and provide mental health insights.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Store user message in chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # If this is the first message, predict the condition
    if not st.session_state.first_prediction_done:
        # Predict mental health condition
        prediction = model.predict([preprocess_text(user_input)])[0]

        # Get condition details
        info = condition_info.get(prediction, {})

        # Generate AI response with classification details
        response = f"Based on your input, I sense **{prediction}**. Here’s some information:\n\n"
        response += f"**Symptoms:** {info.get('symptoms', 'No data available.')}\n\n"
        response += f"**Why is this happening?** {info.get('why', 'No data available.')}\n\n"
        response += f"**How to tackle it?** {info.get('how_to_tackle', 'No data available.')}\n\n"
        response += f"**Steps to take:** {info.get('steps', 'No data available.')}\n\n"
        response += f"**Prevention tips:** {info.get('prevention', 'No data available.')}\n\n"

        # Mark prediction as done
        st.session_state.first_prediction_done = True
    else:
        # If not the first message, only use Gemini AI
        response = get_gemini_response(user_input)

    # Store bot message in chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(response)

