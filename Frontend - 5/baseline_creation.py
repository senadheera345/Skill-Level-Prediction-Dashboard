import os
import pandas as pd
from preprocessing import preprocess_video
import numpy as np

# Paths for each skill level
paths = {
    "Beginner": "C:/Users/Administrator/Desktop/Research/Data/Beginner",
    "Expert": "C:/Users/Administrator/Desktop/Research/Data/Expert",
    "Intermediate": "C:/Users/Administrator/Desktop/Research/Data/Intermediate/Intermediate"
}

def create_baseline_data():
    data = []
    
    for skill_level, path in paths.items():
        for video_file in os.listdir(path):
            video_path = os.path.join(path, video_file)
            hand_motion_data = preprocess_video(video_path)
            
            # Flatten and store the data
            for frame_data in hand_motion_data:
                # Ensure frame_data is a flat list of numbers
                if len(frame_data) == 42:
                    frame_data = list(map(float, frame_data))  # Convert to float for consistency
                    data.append([skill_level] + frame_data)
    
    # Create a DataFrame
    columns = ["SkillLevel"] + [f"Point_{i}" for i in range(1, 43)]
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv("baseline_hand_motion_data.csv", index=False)

def load_baseline_data():
    # Load the baseline data and compute the average
    df = pd.read_csv("baseline_hand_motion_data.csv")
    
    beginner_data = df[df['SkillLevel'] == 'Beginner'].drop('SkillLevel', axis=1).values
    intermediate_data = df[df['SkillLevel'] == 'Intermediate'].drop('SkillLevel', axis=1).values
    expert_data = df[df['SkillLevel'] == 'Expert'].drop('SkillLevel', axis=1).values

    beginner_avg = np.mean(beginner_data, axis=0)
    intermediate_avg = np.mean(intermediate_data, axis=0)
    expert_avg = np.mean(expert_data, axis=0)

    return beginner_avg, intermediate_avg, expert_avg

if __name__ == "__main__":
    create_baseline_data()
