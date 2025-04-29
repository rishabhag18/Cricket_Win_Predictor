
import streamlit as st
import pickle
import sklearn
import pandas as pd
import plotly.express as px





teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))
st.title('🏏 IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))

team2 = [team for team in teams if team != batting_team]

with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(team2))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input("🎯 Target Score", min_value=1, max_value=350, step=1, format="%d")

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input("🏏 Current Score", min_value=0, max_value=target, step=1, format="%d")
    if( score >= target):
        st.toast("Score should be less than target", icon="⚠️")
        st.stop()

with col4:
    overs = st.number_input("⏱ Overs Completed", min_value=1, max_value=20, step=1, format="%d")


with col5:
    wickets = st.number_input("🚨 Wickets Fallen", min_value=0, max_value=10, step=1, format="%d")

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]



    col_result, col_chart = st.columns([1, 1.5])

    with col_result:
        st.subheader("🔮 Win Probability")
        st.write(f"### 🟢 {batting_team}: **{round(win * 100)}%**")
        st.write(f"### 🔴 {bowling_team}: **{round(loss * 100)}%**")

    with col_chart:
        # Pie Chart
        pie_df = pd.DataFrame({
            'Outcome': [f'{batting_team} Win', f'{bowling_team} Win'],
            'Probability': [win, loss]
        })

        fig_pie = px.pie(
            pie_df, values='Probability', names='Outcome',
            color_discrete_sequence=['#00cc96', '#EF553B'],
            title='',
            hole=0.4
        )

        st.plotly_chart(fig_pie, use_container_width=True)





