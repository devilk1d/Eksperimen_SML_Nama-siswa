import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Pastikan resource NLTK sudah tersedia
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def text_preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)               # Hapus HTML
    text = re.sub(r'http\S+|www\S+', '', text)    # Hapus URL
    text = re.sub(r'\d+', '', text)                # Hapus Angka
    text = re.sub(r'[^\w\s]', '', text)           # Hapus Simbol
    text = re.sub(r'\s+', ' ', text).strip()       # Hapus Spasi Ganda
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(clean_tokens)

def preprocess_spam_email_dataset(input_csv, output_csv=None):
    """
    Melakukan preprocessing pada dataset spam email.
    Args:
        input_csv (str): Path ke file CSV mentah.
        output_csv (str, optional): Path file output hasil preprocessing. Jika None, tidak disimpan ke file.
    Returns:
        pd.DataFrame: DataFrame hasil preprocessing, siap dilatih.
    """
    df = pd.read_csv(input_csv)

    # Hapus kolom FILE_NAME jika ada
    if 'FILE_NAME' in df.columns:
        df.drop(columns=['FILE_NAME'], inplace=True)

    # Rename kolom
    df.rename(columns={'CATEGORY': 'label', 'MESSAGE': 'text'}, inplace=True)

    # Mapping label (0=ham, 1=spam)
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df['label_str'] = df['label'].map({0: 'ham', 1: 'spam'})

    # Preprocessing teks
    df['clean_text'] = df['text'].apply(text_preprocessing)

    # Hapus duplikasi pada clean_text
    df.drop_duplicates(subset=['clean_text'], keep='first', inplace=True)

    # Hapus data kosong pada clean_text
    df['clean_text'] = df['clean_text'].replace('', np.nan)
    df.dropna(subset=['clean_text'], inplace=True)

    # Hapus kolom text asli
    if 'text' in df.columns:
        df.drop(columns=['text'], inplace=True)

    # Simpan ke file jika diminta
    if output_csv:
        df.to_csv(output_csv, index=False)

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocessing otomatis dataset spam email.")
    parser.add_argument('input_csv', help='Path ke file CSV mentah')
    parser.add_argument('--output_csv', help='Path file output hasil preprocessing', default=None)
    args = parser.parse_args()
    df_clean = preprocess_spam_email_dataset(args.input_csv, args.output_csv)
    print("Preprocessing selesai. Data siap digunakan.")
    print(df_clean.head())