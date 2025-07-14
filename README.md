# textmining
これは、私が実際の研究で使用しているソースコードです。

import sys
sys.argv = ["analyze.py", "--text", "data/sample_text.txt", "--dict", "data/pn_ja.csv"]
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from janome.tokenizer import Tokenizer
import pandas as pd
import networkx as nx

# テキスト入力（分析対象の文章）
text = """
今日はとても嬉しい日だ。新しいゲームが最高に面白くてハマった！
しかも、友達が優しくて、気分がとてもいい。これは神対応かも。
"""

# ▼ 感情辞書の読み込み（pn_ja.csv に term, value 列が必要）
pn_df = pd.read_csv("pn_ja.csv", encoding="shift_jis", names=["term", "value"], skiprows=1)
pn_dict = dict(zip(pn_df["term"], pn_df["value"]))

# ▼ 形態素解析（形容詞・名詞を抽出、共起分析用に全単語も保持）
tokenizer = Tokenizer()
除外語 = {"ない", "する", "いい"}
words = [
    token.base_form for token in tokenizer.tokenize(text)
    if (token.part_of_speech.startswith("形容詞") or token.part_of_speech.startswith("名詞"))
    and token.base_form not in 除外語
]

# ▼ 出現頻度の集計
word_counts = Counter(words)

# ▼ ワードクラウドの生成（形容詞）
形容詞 = [
    token.base_form for token in tokenizer.tokenize(text)
    if token.part_of_speech.startswith("形容詞") and token.base_form not in 除外語
]
形容詞_counts = Counter(形容詞)

font_path = "C:/Windows/Fonts/meiryo.ttc"
if 形容詞_counts:
    wordcloud = WordCloud(font_path=font_path, background_color='white', width=800, height=400)
    wordcloud.generate_from_frequencies(形容詞_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
else:
    print("⚠️ 抽出された形容詞がありませんでした。")

# ▼ 感情分析（形容詞 + 名詞 × 感情辞書）
emotion_scores = {"positive": 0, "negative": 0}
for word in words:
    score = pn_dict.get(word)
    if score is not None:
        try:
            score = float(score)
            if score > 0:
                emotion_scores["positive"] += score
            elif score < 0:
                emotion_scores["negative"] += abs(score)
        except:
            continue

print("▶️ 感情スコア:", emotion_scores)

# ▼ 共起ネットワークの構築（単純な隣接語2語単位で）
window_size = 2
cooccurrences = []
for i in range(len(words) - window_size + 1):
    window = words[i:i + window_size]
    if len(set(window)) == window_size:
        cooccurrences.append((window[0], window[1]))

cooccurrence_counts = Counter(cooccurrences)

# グラフ構築
G = nx.Graph()
for (word1, word2), count in cooccurrence_counts.items():
    G.add_edge(word1, word2, weight=count)

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, k=0.5, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700)
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_labels(G, pos, font_family="Meiryo", font_size=12)
plt.title("共起ネットワーク", fontsize=16)
plt.axis("off")
plt.show()

# ▼ CSV出力（出現頻度）
df = pd.DataFrame(word_counts.most_common(30), columns=["単語", "出現回数"])
df.to_csv("word_count_result.csv", index=False, encoding="shift_jis")
print(df.head())
