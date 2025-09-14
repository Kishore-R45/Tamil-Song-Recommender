import streamlit as st
from recommender import build_recommender, recommend_song

st.title("🎶 Tamil Song Recommender")
st.write("Pick a Tamil song and get recommendations based on lyrics similarity!")

@st.cache_resource
def load_system():
    return build_recommender()

df, model, index = load_system()

song_list = df['Song Name'].dropna().unique().tolist()
selected_song = st.selectbox("Pick a Tamil song:", song_list)

if st.button("Recommend"):
    best_song, recommendations = recommend_song(selected_song, df, model, index, topn=5)

    if best_song:
        st.write("## ⭐ Most Recommended Song:")
        st.success(f"🎵 {best_song['Song Name']}")
        st.text_area("Tamil Lyrics", best_song['Lyrics'], height=150)

        st.write("## 🎶 Other Similar Songs:")
        for rec in recommendations:
            st.write(f"🎵 {rec['Song Name']}")
            with st.expander("View Lyrics"):
                st.text_area("Tamil", rec['Lyrics'], height=100)
    else:
        st.warning("No similar songs found.")
