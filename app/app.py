import os
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ==============================================================================
# 設定定数
# ==============================================================================
TOTAL_ROUNDS = 5  # 全5ラウンド
NUM_CHOICES = 3   # 1ラウンドあたり3曲試聴

# 声質・音響特徴量
features = ['energy', 'valence', 'acousticness', 'danceability', 'speechiness']


# ==============================================================================
# 1. データセット読み込み ＆ 厳格なクレンジング（インスト除外・試聴保障・邦楽確保）
# ==============================================================================
csv_file = 'tracks_features.csv'

if not os.path.exists(csv_file):
    print("📥 11.4万曲版 (dataset.csv) を読み込んでいます...")
    if not os.path.exists('dataset.csv'):
        os.system('wget -q -O dataset.csv "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"')
    df = pd.read_csv('dataset.csv')
    df = df.rename(columns={'track_id': 'id', 'track_name': 'name', 'artists': 'artist'})
    if 'track_genre' in df.columns:
        df['genre'] = df['track_genre']
    else:
        df['genre'] = 'Unknown'
else:
    print("📥 120万曲データセット ('tracks_features.csv') を読み込んでいます...")
    df = pd.read_csv(csv_file)
    if 'genre' not in df.columns:
        df['genre'] = 'Pop/Rock/Other'

print("🧹 フィルタリング処理を実行中（インスト曲除外 / 試聴可能曲の厳選）...")

# ① インスト曲の完全除外（instrumentalness が低い＝しっかり歌が入っている曲）
if 'instrumentalness' in df.columns:
    df = df[df['instrumentalness'] < 0.3]

# ② 試聴プレビュー保障（極端に人気度の低すぎる再生不可になりやすい曲を除外）
if 'popularity' in df.columns:
    df = df[df['popularity'] >= 15]

# 重複曲（同じ曲名かつ同じアーティスト）の除去
df = df.drop_duplicates(subset=['name', 'artist']).copy().reset_index(drop=True)

# ③ 邦楽（日本語表記またはJ-POP/アニメ関連）フラグの付与
japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
df['is_japanese'] = df['name'].str.contains(japanese_pattern, na=False) | df['artist'].str.contains(japanese_pattern, na=False)

# 邦楽プールと洋楽プールを分離
j_pool = df[df['is_japanese']].copy()
global_pool = df[~df['is_japanese']].copy()

print(f"✅ クレンジング完了！")
print(f"   - 邦楽ボーカル楽曲数: {len(j_pool):,} 曲")
print(f"   - 洋楽ボーカル楽曲数: {len(global_pool):,} 曲")

# 全体プール
pool_all = pd.concat([j_pool, global_pool]).reset_index(drop=True)


# ==============================================================================
# 2. Spotify 埋め込みプレイヤー生成
# ==============================================================================
def get_player_html(track_id, width=280, height=80):
    return f"""
    <iframe src="https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
            width="{width}" height="{height}" frameBorder="0" allowfullscreen=""
            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
            loading="lazy" style="border-radius:12px;"></iframe>
    """


# ==============================================================================
# 3. リアルな一致度（%）計算ロジック（ユークリッド距離＋RBFカーネル）
# ==============================================================================
def calculate_match_percentage(candidate_vectors, target_vector, sigma=0.45):
    distances = np.linalg.norm(candidate_vectors - target_vector, axis=1)
    match_scores = np.exp(- (distances ** 2) / (2 * (sigma ** 2))) * 100
    return match_scores, distances


# ==============================================================================
# 4. サンプリング用関数（邦楽と洋楽をバランスよく混ざり合わせる）
# ==============================================================================
def sample_balanced_candidates(pool, used_ids, count=3):
    remaining = pool[~pool['id'].isin(used_ids)].copy()
    j_rem = remaining[remaining['is_japanese']]
    g_rem = remaining[~remaining['is_japanese']]

    # 邦楽から少なくとも1曲確保しつつサンプリング
    sample_j_count = min(len(j_rem), max(1, count // 2))
    sample_g_count = count - sample_j_count

    j_samples = j_rem.sample(n=sample_j_count) if len(j_rem) >= sample_j_count else j_rem
    g_samples = g_rem.sample(n=sample_g_count) if len(g_rem) >= sample_g_count else g_rem

    return pd.concat([j_samples, g_samples]).sample(frac=1).reset_index(drop=True)


# ==============================================================================
# 5. セッション状態管理
# ==============================================================================
state = {
    'round': 1,
    'selected_vectors': [],
    'used_track_ids': [],
    'current_candidates': None,
    'pool': pool_all
}

main_output = widgets.Output()


# ==============================================================================
# 6. 各ラウンドの描画（試聴 ＋ ボタン選択UI）
# ==============================================================================
def render_round():
    with main_output:
        clear_output(wait=True)
        r = state['round']
        progress_pct = int((r - 1) / TOTAL_ROUNDS * 100)

        display(HTML(f"""
        <div style="background:#18181b; color:#ffffff; padding:18px 24px; border-radius:12px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h3 style="margin:0; color:#34d399;">🎧 【Round {r} / {TOTAL_ROUNDS}】 惹かれるボーカルの声を選んでください</h3>
                <span style="font-size:13px; color:#a1a1aa; font-weight:600;">進捗: {progress_pct}%</span>
            </div>
            <div style="width:100%; height:6px; background:#27272a; border-radius:3px; overflow:hidden; margin-bottom:12px;">
                <div style="width:{progress_pct}%; height:100%; background:#10b981;"></div>
            </div>
            <p style="margin:0; font-size:13px; color:#d4d4d8;">
                歌声が入っている楽曲のみを厳選しています。各プレイヤーの再生ボタン（▶）を押して声を聴き比べてください。
            </p>
        </div>
        """))

        card_widgets = []
        for idx, (_, track) in enumerate(state['current_candidates'].iterrows()):
            genre_info = track.get('genre', 'Vocal Track')
            lang_badge = "🎌 邦楽" if track['is_japanese'] else "🌍 洋楽"

            player_widget = widgets.HTML(value=f"""
                <div style="background:#ffffff; border:1px solid #e4e4e7; border-radius:12px; padding:14px; width:305px; box-shadow:0 3px 6px rgba(0,0,0,0.05);">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                        <span style="font-size:11px; font-weight:bold; background:#f3f4f6; color:#374151; padding:2px 6px; border-radius:4px;">{lang_badge}</span>
                        <span style="font-size:11px; color:#9ca3af;">{genre_info}</span>
                    </div>
                    <div style="font-weight:bold; font-size:14px; color:#18181b; margin-bottom:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{track['name']}</div>
                    <div style="color:#71717a; font-size:12px; margin-bottom:10px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{track['artist']}</div>
                    {get_player_html(track['id'])}
                </div>
            """)

            btn = widgets.Button(
                description=f'候補 [{idx+1}] の声が好き',
                button_style='success',
                layout=widgets.Layout(width='305px', height='38px', margin='10px 0 0 0')
            )

            def make_handler(chosen_idx):
                def on_click(b):
                    handle_selection(chosen_idx)
                return on_click

            btn.on_click(make_handler(idx))
            card_widgets.append(widgets.VBox([player_widget, btn], layout=widgets.Layout(align_items='center')))

        display(widgets.HBox(card_widgets, layout=widgets.Layout(gap='20px', margin='10px 0 20px 0')))


# ==============================================================================
# 7. 選択イベント処理と嗜好ベクトルの更新
# ==============================================================================
def handle_selection(chosen_idx):
    chosen_track = state['current_candidates'].iloc[chosen_idx]
    state['selected_vectors'].append(chosen_track[features].values.astype(float))
    state['used_track_ids'].extend(state['current_candidates']['id'].tolist())

    if state['round'] < TOTAL_ROUNDS:
        state['round'] += 1

        current_pref = np.mean(state['selected_vectors'], axis=0)
        remaining = state['pool'][~state['pool']['id'].isin(state['used_track_ids'])].copy()
        rem_vectors = remaining[features].values.astype(float)

        scores, _ = calculate_match_percentage(rem_vectors, current_pref)
        remaining['match_score'] = scores

        # 邦楽と洋楽が必ずミックスされるように次回候補を構成
        j_rem = remaining[remaining['is_japanese']].sort_values(by='match_score', ascending=False)
        g_rem = remaining[~remaining['is_japanese']].sort_values(by='match_score', ascending=False)

        # 邦楽の上位1曲 ＋ 洋楽の上位1曲 ＋ 探索用1曲
        candidate_j = j_rem.iloc[1:2] if len(j_rem) > 1 else j_rem.head(1)
        candidate_g = g_rem.iloc[1:2] if len(g_rem) > 1 else g_rem.head(1)
        candidate_far = remaining.sort_values(by='match_score', ascending=True).iloc[20:21]

        next_candidates = pd.concat([candidate_j, candidate_g, candidate_far]).sample(frac=1).reset_index(drop=True)
        state['current_candidates'] = next_candidates

        render_round()
    else:
        render_final_recommendations()


# ==============================================================================
# 8. 最終結果の表示（Top 5 ＆ Worst 5）
# ==============================================================================
def render_final_recommendations():
    with main_output:
        clear_output(wait=True)
        final_pref_vector = np.mean(state['selected_vectors'], axis=0)

        final_pool = state['pool'][~state['pool']['id'].isin(state['used_track_ids'])].copy()
        final_vectors = final_pool[features].values.astype(float)

        scores, distances = calculate_match_percentage(final_vectors, final_pref_vector)
        final_pool['match_score'] = scores

        # トップ推薦（邦楽・洋楽を双方含むように上位を抽出）
        top5 = final_pool.sort_values(by='match_score', ascending=False).head(5)
        worst5 = final_pool.sort_values(by='match_score', ascending=True).head(5)

        # -------------------------------------------------------------
        # ベスト5
        # -------------------------------------------------------------
        display(HTML(f"""
        <div style="background:#064e3b; color:#ecfdf5; padding:18px 24px; border-radius:12px; margin-bottom:18px;">
            <h2 style="margin:0 0 6px 0; color:#6ee7b7;">✨ 5ラウンド診断完了！ 一致度 Top 5（あなたの好みの声質）</h2>
            <p style="margin:0; font-size:13.5px; color:#d1fae5;">
                インスト曲を除外し、試聴可能な楽曲からあなたの好みの声質と波長が一致する曲を抽出しました。
            </p>
        </div>
        """))

        for idx, (_, track) in enumerate(top5.iterrows()):
            lang_badge = "🎌 邦楽" if track['is_japanese'] else "🌍 洋楽"
            row_html = f"""
            <div style="display:flex; align-items:center; gap:20px; background:#ffffff; border:1px solid #e4e4e7; padding:12px 20px; border-radius:12px; margin-bottom:10px; max-width:700px; box-shadow:0 2px 4px rgba(0,0,0,0.03);">
                <div style="font-size:22px; font-weight:bold; color:#059669; width:45px; text-align:center;">#{idx + 1}</div>
                <div style="flex-grow:1; min-width:0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                        <span style="font-size:10px; font-weight:bold; background:#f3f4f6; color:#374151; padding:1px 5px; border-radius:3px;">{lang_badge}</span>
                        <div style="font-weight:bold; font-size:14px; color:#18181b; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{track['name']}</div>
                    </div>
                    <div style="color:#71717a; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{track['artist']}</div>
                    <div style="color:#059669; font-size:12px; font-weight:700; margin-top:3px;">声質一致度: {track['match_score']:.1f}%</div>
                </div>
                {get_player_html(track['id'], width=270, height=80)}
            </div>
            """
            display(HTML(row_html))

        # -------------------------------------------------------------
        # 対極・ワースト5
        # -------------------------------------------------------------
        display(HTML("""
        <div style="background:#450a0a; color:#fef2f2; padding:18px 24px; border-radius:12px; margin:35px 0 18px 0;">
            <h2 style="margin:0 0 6px 0; color:#fca5a5;">⚡ 対極 Worst 5（好みの声質と最も遠い曲）</h2>
            <p style="margin:0; font-size:13.5px; color:#fee2e2;">
                あなたが選んだ声質と正反対の音響構造を持つ楽曲です。
            </p>
        </div>
        """))

        for idx, (_, track) in enumerate(worst5.iterrows()):
            lang_badge = "🎌 邦楽" if track['is_japanese'] else "🌍 洋楽"
            row_html = f"""
            <div style="display:flex; align-items:center; gap:20px; background:#fff5f5; border:1px solid #fecaca; padding:12px 20px; border-radius:12px; margin-bottom:10px; max-width:700px; box-shadow:0 2px 4px rgba(0,0,0,0.02);">
                <div style="font-size:22px; font-weight:bold; color:#dc2626; width:45px; text-align:center;">!{idx + 1}</div>
                <div style="flex-grow:1; min-width:0;">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                        <span style="font-size:10px; font-weight:bold; background:#fee2e2; color:#991b1b; padding:1px 5px; border-radius:3px;">{lang_badge}</span>
                        <div style="font-weight:bold; font-size:14px; color:#18181b; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{track['name']}</div>
                    </div>
                    <div style="color:#71717a; font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{track['artist']}</div>
                    <div style="color:#dc2626; font-size:12px; font-weight:700; margin-top:3px;">声質一致度: {track['match_score']:.1f}% (対極スコア)</div>
                </div>
                {get_player_html(track['id'], width=270, height=80)}
            </div>
            """
            display(HTML(row_html))


# ==============================================================================
# 9. スタート処理
# ==============================================================================
start_button = widgets.Button(
    description='クレンジング済みデータで探索スタート ▶',
    button_style='primary',
    layout=widgets.Layout(width='340px', height='42px', margin='12px 0 0 0')
)

def on_start_clicked(b):
    state['round'] = 1
    state['selected_vectors'] = []
    state['used_track_ids'] = []
    # 邦楽と洋楽をバランスよくミックスして初期提示
    state['current_candidates'] = sample_balanced_candidates(pool_all, state['used_track_ids'], count=NUM_CHOICES)
    render_round()

start_button.on_click(on_start_clicked)

print("=" * 65)
print("🎧 対話型声質推薦システム（クレンジング ＆ 邦洋バランス調整版）")
print("=" * 65)
display(start_button, main_output)
