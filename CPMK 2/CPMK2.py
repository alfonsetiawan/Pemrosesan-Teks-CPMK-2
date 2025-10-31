import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk

# --- Unduh resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# --- Baca file hasil scraping
df = pd.read_csv("youtube_comments.csv")

# --- Kamus normalisasi kata tidak baku
normalisasi = {
    "gk": "tidak", "ga": "tidak", "nggak": "tidak", "ngga": "tidak",
    "bgt": "banget", "bngt": "banget", "bener": "benar",
    "yg": "yang", "dgn": "dengan", "sm": "sama", "aja": "saja",
    "kok": "", "nih": "", "dong": "", "deh": "", "loh": "", "lah": "",
    "tp": "tapi", "jd": "jadi", "krn": "karena", "dr": "dari", "trs": "terus"
}

# --- Fungsi preprocessing
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()                                   # lowercase
    text = re.sub(r"http\S+|www\S+", "", text)            # hapus URL
    text = re.sub(r"@\w+|#\w+", "", text)                 # hapus mention & hashtag
    text = text.translate(str.maketrans("", "", string.punctuation))  # hapus tanda baca
    text = re.sub(r"\d+", "", text)                       # hapus angka
    text = text.encode('ascii', 'ignore').decode('ascii') # hapus emoji
    text = " ".join([normalisasi.get(w, w) for w in text.split()])  # normalisasi
    stop_words = set(stopwords.words('indonesian'))
    text = " ".join([w for w in text.split() if w not in stop_words]) # hapus stopword
    return text.strip()

# --- Bersihkan komentar
df['cleaned'] = df['comment'].astype(str).apply(clean_text)

# --- Hapus duplikat & kosong
df = df.drop_duplicates(subset=['cleaned'])
df = df[df['cleaned'].str.len() > 0]

# --- Buat Bigram & Trigram
def make_ngrams(text, n):
    tokens = word_tokenize(text)
    return [" ".join(g) for g in ngrams(tokens, n)]

df['bigram'] = df['cleaned'].apply(lambda x: make_ngrams(x, 2))
df['trigram'] = df['cleaned'].apply(lambda x: make_ngrams(x, 3))

# --- WordCloud
all_words = " ".join(df['cleaned'])
wordcloud = WordCloud(width=1200, height=800, background_color="white").generate(all_words)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud Komentar YouTube", fontsize=16)
plt.show()

# --- Simpan hasil akhir (3 kolom saja)
df[['cleaned', 'bigram', 'trigram']].to_csv("youtube_comments_processed.csv", index=False, encoding='utf-8-sig')

print("✅ File preprocessing selesai → youtube_comments_processed.csv")
