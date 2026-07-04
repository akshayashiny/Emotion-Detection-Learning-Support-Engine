import google.generativeai as genai

from config import GEMINI_API_KEY

# -----------------------
# Configure Gemini
# -----------------------

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


def generate_response(user_input, emotion, confidence, field):
    """
    Generate an empathetic AI response using Gemini.
    Falls back to a template if Gemini fails.
    """

    prompt = f"""
You are an empathetic AI learning assistant.

Student Subject:
{field}

Detected Emotion:
{emotion}

Confidence:
{confidence*100:.2f}%

Student Problem:
{user_input}

Instructions:
1. Be supportive and encouraging.
2. Explain why the student may feel this way.
3. Give practical study advice specific to the subject.
4. Keep the response under 200 words.
5. End with one motivational sentence.
"""

    try:
        response = model.generate_content(prompt)

        if response.text:
            return response.text

    except Exception:
        pass

    # -----------------------
    # Fallback Response
    # -----------------------

    return f"""
Emotion Detected: {emotion}

It is completely normal to feel {emotion.lower()} while learning {field}.

Suggestions:
• Revise the basics.
• Practice one concept at a time.
• Take short breaks.
• Ask questions whenever you're stuck.
• Stay consistent.

Keep going—you'll improve with practice.
"""