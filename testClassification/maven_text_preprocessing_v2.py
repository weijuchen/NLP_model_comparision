# import necessary libraries
import pandas as pd
import spacy
import swifter

# download the spacy model
nlp = spacy.load("en_core_web_sm")

# helper functions from text preprocessing section
# def lower_replace(series):   # 僅能輸入參數  pandas series


#     output = series.str.lower()
#     # output = output.str.replace(r'\[.*?\]', '', regex=True)
#     output = output.str.replace(r'[^\w\s]', '', regex=True)  # remove punctuation
#     return output

def token_lemma_nonstop(text):    #  僅能丟入參數 是series 中的單筆資料
    doc = nlp(text)
    output = [token.lemma_ for token in doc if not token.is_stop]
    return ' '.join(output)

def clean_and_normalize(series):
    output = series.str.lower()
    output = output.str.replace(r"[^\w\s]", " ", regex=True)
    output=output.str.replace(r"\s+", " ", regex=True) 



    # step 1 remove punctuation, whitespace and lowercase the text
    # output = (
    #     series.str.lower()
    #     .str.replace(r"[^\w\s]+", " ", regex=True)  # remove punctuation
    #     .str.replace(r"\s+", " ", regex=True) # remove spaces
    #     .str.strip() #remove leading and trailing spaces
    # )

    # 步驟 1: 執行 lower_replace
    # output = series.str.lower()
    # output = output.str.replace(r'[^\w\s]', ' ', regex=True)

    # 步驟 2: 使用 swifter 並行處理 NLP 任務
    output = output.swifter.apply(token_lemma_nonstop)
    return output

    # 以下版本慢
    # output = lower_replace(series)
    # output = output.apply(token_lemma_nonstop) # 使用 pandas 的 .apply() 方法來對 Series 中的每個元素（每個文本字串）應用此函數
    # return output

# allow command-line execution
if __name__ == "__main__":
    print("Text preprocessing module ready to use.")
