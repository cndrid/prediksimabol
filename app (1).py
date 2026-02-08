import streamlit as st
import pandas as pd
import joblib

# LOAD MODEL
model = joblib.load("parlay_model.pkl")
columns = joblib.load("columns.pkl")

# PAGE CONFIG
st.set_page_config(
    page_title="AI Parlay Predictor",
    page_icon="⚽",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(180deg,#0c1c2c,#061018);
    color:white;
}

/* HEADER */
.header {
    font-size:40px;
    font-weight:bold;
    text-align:center;
    background: linear-gradient(90deg,#1e90ff,#00bfff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.06);
    padding:25px;
    border-radius:15px;
    backdrop-filter: blur(6px);
}

/* BUTTON */
.stButton>button {
    width:100%;
    height:3em;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
    background: linear-gradient(90deg,#1e90ff,#00bfff);
    color:white;
    transition:0.3s;
}

.stButton>button:hover {
    transform:scale(1.05);
    box-shadow:0 0 20px #00bfff;
}

/* RESULT BOX */
.result {
    font-size:22px;
    padding:20px;
    border-radius:12px;
    background: rgba(0,191,255,0.15);
    text-align:center;
}

/* PROGRESS BAR */
.stProgress > div > div > div {
    background-color:#00bfff;
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("<div class='header'>⚽ AI Football Parlay Predictor</div>", unsafe_allow_html=True)

st.write("### Smart betting insights powered by Machine Learning")

# TEAM INPUT
st.write("## 🏟️ Match Setup")

colA, colB = st.columns(2)

with colA:
    home_team = st.text_input("Home Team","Manchester City")
    HS = st.slider("Home Shots",0,30,12)
    HST = st.slider("Home Shots on Target",0,15,6)
    B365H = st.number_input("Home Odds",1.0,10.0,2.0)

with colB:
    away_team = st.text_input("Away Team","Liverpool")
    AS = st.slider("Away Shots",0,30,9)
    AST = st.slider("Away Shots on Target",0,15,4)
    B365A = st.number_input("Away Odds",1.0,15.0,3.5)

# AUTO TEAM LOGO (via Clearbit logo API)
logo1 = f"https://logo.clearbit.com/{home_team.replace(' ','')}.com"
logo2 = f"https://logo.clearbit.com/{away_team.replace(' ','')}.com"

col1,col2,col3 = st.columns([1,1,1])

with col1:
    st.image(logo1,width=80)

with col3:
    st.image(logo2,width=80)

# PREDICTION
if st.button("🚀 Predict Match"):

    data = pd.DataFrame({
        "HS":[HS],
        "AS":[AS],
        "HST":[HST],
        "AST":[AST],
        "ProbH":[1/B365H],
        "ProbA":[1/B365A],
        "ProbD":[0.25]  # default draw prob placeholder
    })

    data = data.reindex(columns=columns,fill_value=0)

    probs = model.predict_proba(data)[0]

    st.write("## 📊 AI Prediction")

    st.write("Home Win Probability")
    st.progress(float(probs[2]))

    st.write("Draw Probability")
    st.progress(float(probs[1]))

    st.write("Away Win Probability")
    st.progress(float(probs[0]))

    # VALUE BET CHECK
    value = probs[2] > (1/B365H)

    if value:
        st.success("🔥 VALUE BET FOUND on Home Win!")
    else:
        st.warning("No strong value bet")


