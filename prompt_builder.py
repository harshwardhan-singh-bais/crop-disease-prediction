def build_prompt(user_input, disease_json, history):

    context = f"""
You are an AI agricultural assistant for farmers.

STRICT RULES:
- Respond in SAME language as user
- Keep answer simple and practical
- Max 4–5 lines
- No technical jargon

CONTEXT DATA:
Disease: {disease_json.get("disease")}
Crop: {disease_json.get("crop")}
Confidence: {disease_json.get("confidence")}%
Severity: {disease_json.get("severity")}
First Aid: {disease_json.get("first_aid")}
Action Plan: {disease_json.get("action_plan")}
Weather Advice: {disease_json.get("weather_advice")}

"""

    messages = []

    messages.append({
        "role": "system",
        "content": context.strip(),
    })

    # Add history
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
        })

    # Add current query
    messages.append({
        "role": "user",
        "content": user_input,
    })

    return messages