import streamlit as st

def load_ui():
    st.markdown("""
        <style>
        body {
            background-color: #0e1117;
        }

        .main-title {
            font-size: 42px;
            font-weight: 800;
            text-align: center;
            color: #4da3ff;
            margin-bottom: 5px;
        }

        .sub-title {
            text-align: center;
            color: #aab0b6;
            margin-bottom: 40px;
        }

        .card {
            background: #161b22;
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.4);
        }

        .card h3 {
            color: #58a6ff;
            margin-bottom: 10px;
        }

        .card p {
            color: #c9d1d9;
            font-size: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-title">🩺 Healthcare Risk Predictor</div>
        <div class="sub-title">AI-powered patient test result prediction</div>
    """, unsafe_allow_html=True)


def section(title, text):
    st.markdown(f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{text}</p>
        </div>
    """, unsafe_allow_html=True)
def info_block(title, points):
    st.markdown(f"""
        <div class="card">
            <h3>{title}</h3>
            <ul style="color:#c9d1d9; font-size:15px;">
                {''.join([f"<li>{p}</li>" for p in points])}
            </ul>
        </div>
    """, unsafe_allow_html=True)
