class ChatMemory:
    def __init__(self):
        self.sessions = {}

    def get_history(self, session_id):
        return self.sessions.get(session_id, [])

    def update(self, session_id, role, content):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "role": role,
            "content": content
        })

        # Keep last 6 messages
        if len(self.sessions[session_id]) > 6:
            self.sessions[session_id] = self.sessions[session_id][-6:]