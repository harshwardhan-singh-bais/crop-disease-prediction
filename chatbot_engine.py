from memory import ChatMemory
from prompt_builder import build_prompt
from gemini_client import get_response

memory = ChatMemory()


def chatbot_reply(session_id, user_input, disease_json):

    history = memory.get_history(session_id)

    messages = build_prompt(user_input, disease_json, history)

    reply = get_response(messages)

    # Update memory
    memory.update(session_id, "user", user_input)
    memory.update(session_id, "assistant", reply)

    return {
        "reply": reply,
        "language": "auto",
        "session_id": session_id,
    }