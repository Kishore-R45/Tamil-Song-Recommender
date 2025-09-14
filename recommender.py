import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

_tamil_only_re = re.compile(r'[^\u0B80-\u0BFF\s]')

def clean_lyrics(text):
    if pd.isna(text):
        return ""
    s = str(text)
    s = re.sub(r'\[.*?\]', '', s)         
    s = _tamil_only_re.sub('', s)         
    s = re.sub(r'\s+', ' ', s).strip()    
    return s

def build_recommender(data_path="dataset/tamil_songs.csv"):

    df = pd.read_csv(data_path)

    df['clean_lyrics'] = df['Lyrics'].astype(str).apply(clean_lyrics)

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    embeddings = model.encode(df['clean_lyrics'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return df, model, index

def recommend_song(song_title, df, model, index, topn=5):
    if song_title not in df['Song Name'].values:
        return None, []

    song_idx = df[df['Song Name'] == song_title].index[0]
    query_vec = model.encode([df.loc[song_idx, 'clean_lyrics']])

    distances, indices = index.search(np.array(query_vec).astype("float32"), topn+1)

    recs = []
    for i in indices[0][1:]:
        recs.append(df.loc[i, ['Song Name', 'Lyrics']].to_dict())
        if len(recs) >= topn:
            break

    most_recommended = recs[0] if recs else None
    other_recs = recs[1:] if len(recs) > 1 else []

    return most_recommended, other_recs
