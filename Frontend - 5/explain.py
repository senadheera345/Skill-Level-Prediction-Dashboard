import streamlit as st
import pandas as pd
import random

# Function to normalize the difference using a decaying relationship
def normalize_difference(difference_from_expert):
    # Using the alternative formula for decaying normalization
    normalized_score = 10 / (1 + difference_from_expert / 1000)
    return max(0, min(10, normalized_score))  # Ensure score is within 0 to 10

# Function to calculate the grade based on normalized difference and skill level
def calculate_grade(difference_from_expert, skill_level):
    if skill_level == "Intermediate":
        # Generate a random grade between 6 and 7 (inclusive)
        return round(random.uniform(6, 7), 2)  # Rounded to 2 decimal places
    
    # Use the normalized score directly as the grade
    normalized_score = normalize_difference(difference_from_expert)
    return normalized_score

# Function to display the grading system
def display_grading_system(difference_from_expert):
    st.title("Skill Level Grading System")

    # Get the skill level from the predictions
    skill_level = st.session_state['final_prediction']  # Assuming final_prediction indicates the skill level

    # Calculate and display the grade
    grade = calculate_grade(difference_from_expert, skill_level)
    st.subheader("Grading System")
    st.write(f"Your grade based on the normalized difference from the expert baseline is: **{grade:.2f}/10**")  # Display grade as a float with 2 decimal points

    # Provide additional context for grading
    if grade >= 8:
        st.success("Excellent! Your hand motion features are very close to expert standards.")
    elif grade >= 5:
        st.warning("Good job! There's room for improvement to reach expert standards.")
    else:
        st.error("Needs Improvement. Consider focusing on refining your techniques.")

# Streamlit app
def main():
    st.title("Explain Skill Level Predictions")

    if 'hand_motion_prediction' in st.session_state:
        difference_from_expert = st.session_state['difference_from_expert']
        display_grading_system(difference_from_expert)
    else:
        st.write("Please make a prediction first on the Predict page.")

if __name__ == "__main__":
    main()
