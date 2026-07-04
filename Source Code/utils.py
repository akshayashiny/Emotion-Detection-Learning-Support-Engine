from datetime import datetime


def create_report(
    subject,
    model,
    emotion,
    confidence,
    recommendation,
    ai_response
):

    report = f"""
==========================================
Emotion Detection & Learning Support Engine
==========================================

Date:
{datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

------------------------------------------

Subject:
{subject}

Model:
{model}

Detected Emotion:
{emotion}

Confidence:
{confidence*100:.2f}%

------------------------------------------

Learning Recommendation

{recommendation}

------------------------------------------

AI Mentor Guidance

{ai_response}

==========================================
"""

    return report