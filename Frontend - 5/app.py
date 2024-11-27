import streamlit as st

# Function to navigate to different pages
def main():

    st.sidebar.image("C:/Users/Administrator/Desktop/Research/Frontend - 5/Maslogo.png", use_column_width=True)
    st.sidebar.title("LineaFlex")
    page = st.sidebar.radio("Go to", ["Predict", "Explain"])

    if page == "Predict":
        import predict
        predict.main()
    elif page == "Explain":
        import explain
        explain.main()

if __name__ == "__main__":
    main()
