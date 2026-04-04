from chatbot_engine import chatbot_reply
import json

# Sample disease data as would be provided by the intelligence engine
sample_disease_json = {
    "disease": "Tomato Late Blight",
    "crop": "Tomato",
    "confidence": 94,
    "severity": "High",
    "first_aid": "Remove infected leaves immediately and improve air circulation.",
    "action_plan": "Apply fungicides containing copper or chlorothalonil. Avoid overhead watering.",
    "weather_advice": "High humidity favors the spread. Keep the foliage dry."
}

def test_chatbot():
    session_id = "test_user_123"
    
    print("--- Chatbot Test Session ---")
    
    # Test Query 1
    user_input_1 = "What should I do first?"
    print(f"\nUser: {user_input_1}")
    result_1 = chatbot_reply(session_id, user_input_1, sample_disease_json)
    print(f"Assistant: {result_1['reply']}")
    
    # Test Query 2 (Testing memory)
    user_input_2 = "Can I use any natural sprays?"
    print(f"\nUser: {user_input_2}")
    result_2 = chatbot_reply(session_id, user_input_2, sample_disease_json)
    print(f"Assistant: {result_2['reply']}")

if __name__ == "__main__":
    try:
        test_chatbot()
    except Exception as e:
        print(f"Error during test: {e}")
