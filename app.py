import streamlit as st
import pandas as pd
import numpy as np
import os

# ==============================================================================
# ページ初期設定
# ==============================================================================
st.set_page_config(
    page_title="声質対話型音楽推薦システム",
    page_icon="🎧",
    layout="wide"
)

TOTAL_ROUNDS = 5
NUM_CHOICES = 3
FEATURES = ['energy', 'valence', 'acousticness', 'danceability', 'speechiness']

# ==============================================================================
# 1. データロード＆キャッシュ化（URL直読み＆自動フォールバック対応）
# ==============================================================================
@st.cache_data
def load_and_clean_data():
    csv_file = 'tracks_features.csv'
    
    # 優先順位：120万曲ファイル -> dataset.csv -> オンライン直接取得（HuggingFace）
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    elif os.path.exists('dataset.csv'):
        df = pd.read_csv('dataset.csv')
    else:
        # オンライン上のデータセットを取得（FileNotFoundErrorを100%防ぎます）
        url = 'https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv'
        df = pd.read_csv(url)

    # カラム名合わせ
    if 'track_id' in df.columns:
        df = df.rename(columns={'track_id': 'id', 'track_name': 'name', 'artists': 'artist'})
    if 'genre' not in df.columns and 'track_genre' in df.columns:
        df['genre'] = df['track_genre']
    if 'genre' not in df.columns:
        df['genre'] = 'Pop/Rock/Other'

    # クレンジング処理
    if 'instrumentalness' in df.columns:
        df = df[df['instrumentalness'] < 0.3]  # インスト除外
    if 'popularity' in df.columns:
        df = df[df['popularity'] >= 15]        # 試聴不可対策

    df = df.drop_duplicates(subset=['name', 'artist']).copy().reset_index(drop=True)

    # 邦楽判定
    japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
    df['is_japanese'] = df['name'].str.contains(japanese_pattern, na=False) | df['artist'].str.contains(japanese_pattern, na=False)
    
    return df

df_pool = load_and_clean_data()

# ==============================================================================
# 2. ロジック関数（計算・サンプリング）
# ==============================================================================
def calculate_match_percentage(candidate_vectors, target_vector, sigma=0.45):
    distances = np.linalg.norm(candidate_vectors - target_vector, axis=1)
    match_scores = np.exp(- (distances ** 2) / (2 * (sigma ** 2))) * 100
    return match_scores

def sample_balanced_candidates(pool, used_ids, count=3):
    remaining = pool[~pool['id'].isin(used_ids)].copy()
    j_rem = remaining[remaining['is_japanese']]
    g_rem = remaining[~remaining['is_japanese']]
    
    sample_j_count = min(len(j_rem), max(1, count // 2))
    sample_g_count = count - sample_j_count
    
    j_samples = j_rem.sample(n=sample_j_count) if len(j_rem) >= sample_j_count else j_rem
    g_samples = g_rem.sample(n=sample_g_count) if len(g_rem) >= sample_g_count else g_rem
    
    return pd.concat([j_samples, g_samples]).sample(frac=1).reset_index(drop=True)

def render_spotify_player(track_id, height=80):
    iframe_html = f"""
    <iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0" 
            width="100%" height="{height}" frameBorder="0" allowfullscreen="" 
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" 
            loading="lazy" style="border-radius:12px;"></iframe>
    """
    st.components.v1.html(iframe_html, height=height+10)

# ==============================================================================
# 3. セッション状態の初期化
# ==============================================================================
if 'started' not in st.session_state:
    st.session_state.started = False
    st.session_state.round = 1
    st.session_state.selected_vectors = []
    st.session_state.used_track_ids = []
    st.session_state.current_candidates = None

# ==============================================================================
# 4. メインUI表示
# ==============================================================================
st.title("🎧 声質対話型 音楽推薦システム")
st.caption("直感的に惹かれるボーカルの声を選んでいくことで、データベースからあなたの潜在的な声の好みを診断します。")

# --- スタート画面 ---
if not st.session_state.started:
    st.markdown("---")
    st.info("💡 各ラウンドで3つのボーカル楽曲を試聴し、最も「惹かれる声・質感」の曲を1つ選んでください（全5ラウンド）。")
    if st.button("🚀 診断をスタートする", type="primary", use_container_width=True):
        st.session_state.started = True
        st.session_state.round = 1
        st.session_state.selected_vectors = []
        st.session_state.used_track_ids = []
        st.session_state.current_candidates = sample_balanced_candidates(df_pool, [], count=NUM_CHOICES)
        st.rerun()

# --- 診断中（ラウンド 1〜5）---
elif st.session_state.round <= TOTAL_ROUNDS:
    r = st.session_state.round
    progress_val = (r - 1) / TOTAL_ROUNDS
    st.progress(progress_val, text=f"進捗: Round {r} / {TOTAL_ROUNDS}")
    
    st.subheader(f"【Round {r}】 惹かれるボーカルの声を選んでください")
    
    cols = st.columns(NUM_CHOICES)
    candidates = st.session_state.current_candidates
    
    for idx in range(len(candidates)):
        track = candidates.iloc[idx]
        lang_label = "🎌 邦楽" if track['is_japanese'] else "🌍 洋楽"
        
        with cols[idx]:
            st.markdown(f"**候補 [{idx+1}]** `{lang_label}`")
            st.markdown(f"**{track['name']}**")
            st.caption(f"{track['artist']} | {track.get('genre', 'Vocal')}")
            
            render_spotify_player(track['id'])
            
            if st.button(f"この声が好き 👍", key=f"btn_{r}_{idx}", use_container_width=True):
                chosen_vector = track[FEATURES].values.astype(float)
                st.session_state.selected_vectors.append(chosen_vector)
                st.session_state.used_track_ids.extend(candidates['id'].tolist())
                
                if st.session_state.round < TOTAL_ROUNDS:
                    st.session_state.round += 1
                    current_pref = np.mean(st.session_state.selected_vectors, axis=0)
                    rem = df_pool[~df_pool['id'].isin(st.session_state.used_track_ids)].copy()
                    rem_vectors = rem[FEATURES].values.astype(float)
                    
                    scores = calculate_match_percentage(rem_vectors, current_pref)
                    rem['match_score'] = scores
                    
                    j_rem = rem[rem['is_japanese']].sort_values(by='match_score', ascending=False)
                    g_rem = rem[~rem['is_japanese']].sort_values(by='match_score', ascending=False)
                    
                    cand_j = j_rem.iloc[1:2] if len(j_rem) > 1 else j_rem.head(1)
                    cand_g = g_rem.iloc[1:2] if len(g_rem) > 1 else g_rem.head(1)
                    cand_far = rem.sort_values(by='match_score', ascending=True).iloc[20:21]
                    
                    st.session_state.current_candidates = pd.concat([cand_j, cand_g, cand_far]).sample(frac=1).reset_index(drop=True)
                    st.rerun()
                else:
                    st.session_state.round += 1
                    st.rerun()

# --- 最終結果画面 ---
else:
    st.balloons()
    st.success("🎉 5ラウンドの診断が完了しました！")
    
    final_pref_vector = np.mean(st.session_state.selected_vectors, axis=0)
    final_pool = df_pool[~df_pool['id'].isin(st.session_state.used_track_ids)].copy()
    final_vectors = final_pool[FEATURES].values.astype(float)
    
    scores = calculate_match_percentage(final_vectors, final_pref_vector)
    final_pool['match_score'] = scores
    
    top5 = final_pool.sort_values(by='match_score', ascending=False).head(5)
    worst5 = final_pool.sort_values(by='match_score', ascending=True).head(5)
    
    tab1, tab2 = st.tabs(["✨ 一致度 Top 5（好みの声質）", "⚡ 対極 Worst 5（真逆の声質）"])
    
    with tab1:
        st.markdown("### あなたの好みの声質とマッチした楽曲")
        for idx, (_, track) in enumerate(top5.iterrows()):
            lang = "🎌 邦楽" if track['is_japanese'] else "🌍 洋楽"
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"#### #{idx+1} {track['name']}")
                st.write(f"**{track['artist']}** ({lang})")
                st.metric("声質一致度", f"{track['match_score']:.1f}%")
            with c2:
                render_spotify_player(track['id'])
            st.divider()

    with tab2:
        st.markdown("### あなたの好みの声質と正反対の楽曲")
        for idx, (_, track) in enumerate(worst5.iterrows()):
            lang = "🎌 邦楽" if track['is_japanese'] else "🌍 洋楽"
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"#### !{idx+1} {track['name']}")
                st.write(f"**{track['artist']}** ({lang})")
                st.metric("一致度（対極）", f"{track['match_score']:.1f}%")
            with c2:
                render_spotify_player(track['id'])
            st.divider()

    if st.button("🔄 もう一度診断する", use_container_width=True):
        st.session_state.started = False
        st.rerun()
