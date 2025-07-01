import re
import pandas as pd
import os
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|\@\w+|\#\w+', ' ', text)  
    text = re.sub(r'[^a-z0-9\s]', ' ', text)          
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_data():
    raw_data = pd.read_csv('../data/raw/track-b.csv', encoding = 'unicode_escape')
    text_column = 'text'
    # clean just the text column
    raw_data[text_column] = raw_data[text_column].astype(str).apply(clean_text)
    out_dir = '../data/processed/'
    os.makedirs(out_dir, exist_ok=True)
    # 4. Write cleaned CSV
    out_path = os.path.join(out_dir, 'track-b-clean.csv')
    raw_data.to_csv(out_path, index=False, encoding='utf-8')
    
    
    

# if __name__ == "__main__":
#     me =   clean_text("Richard is here Hell ow are you Doing ? # 4 hahaha  I am doing great ")
#     print(me)
#     clean_data();