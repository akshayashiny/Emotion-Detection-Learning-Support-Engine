def get_recommendation(emotion):

    recommendations = {

        "Confused": """
📘 Recommendation:
• Break the topic into smaller concepts.
• Watch one beginner-friendly YouTube video.
• Solve one simple example before harder ones.
• Revise the basics first.
""",

        "Confident": """
🎯 Recommendation:
• Excellent! Try more challenging problems.
• Practice consistently to strengthen your skills.
• Help someone else understand the topic.
""",

        "Curious": """
🔍 Recommendation:
• Explore the topic in greater depth.
• Read additional articles or documentation.
• Experiment with small projects.
""",

        "Frustrated": """
😌 Recommendation:
• Take a short break.
• Return with a fresh mind.
• Solve easier questions first.
• Ask for help if you're stuck.
""",

        "Bored": """
🚀 Recommendation:
• Try interactive coding platforms.
• Watch engaging tutorials.
• Set yourself a small challenge.
""",

        "Neutral": """
📚 Recommendation:
• Continue practicing consistently.
• Review today's concepts.
""",

        "Happy": """
🎉 Recommendation:
• Keep up the momentum.
• Try a more difficult problem.
""",

        "Sad": """
💙 Recommendation:
• Take a short break.
• Don't compare yourself with others.
• Start again with a small goal.
""",

        "Anxious": """
🌱 Recommendation:
• Focus on one topic at a time.
• Take deep breaths.
• Avoid multitasking.
"""
    }

    return recommendations.get(
        emotion,
        "📚 Keep learning consistently and practice every day."
    )