import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from collections import Counter
import os
from preprocessing import preprocess_video
import concurrent.futures

@st.cache_resource
def load_models():
    hand_model = load('trained_model_strsmlit.joblib')
    demographic_model = load('gradient_boosting_model_95%.joblib')
    return hand_model, demographic_model

hand_model, demographic_model = load_models()

default_directory = "C:/Users/Administrator/Desktop/Research/Data/test"
os.makedirs(default_directory, exist_ok=True)

def load_baseline_data():
    df = pd.read_csv("baseline_hand_motion_data.csv")
    beginner_data = df[df['SkillLevel'] == 'Beginner'].drop('SkillLevel', axis=1).values
    intermediate_data = df[df['SkillLevel'] == 'Intermediate'].drop('SkillLevel', axis=1).values
    expert_data = df[df['SkillLevel'] == 'Expert'].drop('SkillLevel', axis=1).values

    beginner_avg = np.mean(beginner_data, axis=0)
    intermediate_avg = np.mean(intermediate_data, axis=0)
    expert_avg = np.mean(expert_data, axis=0)

    return beginner_avg, intermediate_avg, expert_avg

beginner_avg, intermediate_avg, expert_avg = load_baseline_data()

def predict_skill_level_from_hand_motion(hand_motion_data):
    if len(hand_motion_data) == 0:
        return None  # Indicate that no hand motion data was found
    hand_motion_df = pd.DataFrame(hand_motion_data)
    predictions = hand_model.predict(hand_motion_df)
    prediction_counts = Counter(predictions)
    majority_prediction = prediction_counts.most_common(1)[0][0]
    return majority_prediction

def predict_skill_level_from_demographic(age, gender, avg_working_hours, experienced_time):
    prediction = demographic_model.predict([[age, gender, avg_working_hours, experienced_time]])[0]
    return prediction

def calculate_difference_from_expert(hand_motion_data):
    hand_motion_data_np = np.array(hand_motion_data).mean(axis=0)
    expert_avg_np = np.array(expert_avg)
    difference = np.linalg.norm(hand_motion_data_np - expert_avg_np)
    return difference

def main():
    st.title("Skill Level Prediction Dashboard")
    st.sidebar.header("Upload Video and Enter Demographics")

    st.sidebar.subheader("Hand Motion Prediction")
    video_file = st.sidebar.file_uploader("Upload video", type=["mp4"])

    st.sidebar.subheader("Demographic Information")
    age = st.sidebar.slider("Age", 18, 35, 23)
    gender = st.sidebar.radio("Gender", ['Male', 'Female'])
    gender = 1 if gender == 'Male' else 0
    avg_working_hours = st.sidebar.slider("Average Working Hours", 0.0, 40.0, 30.0)
    experienced_time = st.sidebar.slider("Experienced Time", 0.0, 5.0, 1.5)

    if st.sidebar.button("Predict Skill Level"):
        hand_motion_prediction = None
        demographic_prediction = None
        difference_from_expert = None

        def process_video():
            temp_video_path = os.path.join(default_directory, video_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())

            hand_motion_data = preprocess_video(temp_video_path)

            if len(hand_motion_data) == 0:
                os.remove(temp_video_path)
                return "Upload a valid video"  # No hand motion data detected

            prediction = predict_skill_level_from_hand_motion(hand_motion_data)
            nonlocal difference_from_expert
            difference_from_expert = calculate_difference_from_expert(hand_motion_data)
            os.remove(temp_video_path)
            return prediction

        if video_file:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                video_future = executor.submit(process_video)
        
        demographic_prediction = predict_skill_level_from_demographic(age, gender, avg_working_hours, experienced_time)

        if video_file:
            hand_motion_prediction = video_future.result()
            if hand_motion_prediction == "Upload a valid video":
                st.error("Upload a valid video")
                return

        st.subheader("Predictions")
        if hand_motion_prediction is not None:
            st.write(f"Hand Motion Prediction: <span style='color:green'><strong>{hand_motion_prediction}</strong></span>", unsafe_allow_html=True)
        st.write(f"Demographic Prediction: <span style='color:green'><strong>{demographic_prediction}</strong></span>", unsafe_allow_html=True)

        if hand_motion_prediction is not None:
            final_predictions = [hand_motion_prediction, demographic_prediction]
            final_prediction = Counter(final_predictions).most_common(1)[0][0]
        else:
            final_prediction = demographic_prediction

        st.subheader("Final Skill Level Prediction")
        st.write(f"<span style='color:red'><strong>{final_prediction}</strong></span>", unsafe_allow_html=True)

        st.subheader("Difference from Expert Baseline")
        st.write(f"The calculated difference from the expert baseline is <span style='color:orange'><strong>{difference_from_expert:.2f}</strong></span>.", unsafe_allow_html=True)

        st.markdown(
    """
    A <span style='color:green;'>lower value</span> indicates that your hand motion features are closer to those of an expert. 
    A <span style='color:red;'>higher value</span> suggests that there is a greater difference between your hand motion features and those of an expert.
    """, unsafe_allow_html=True)
        
        st.session_state['hand_motion_prediction'] = hand_motion_prediction
        st.session_state['demographic_prediction'] = demographic_prediction
        st.session_state['final_prediction'] = final_prediction
        st.session_state['difference_from_expert'] = difference_from_expert

if __name__ == "__main__":
    main()
