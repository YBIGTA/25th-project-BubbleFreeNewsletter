import os
import json
import mysql.connector
from mysql.connector import Error
import configparser
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn

'''
모델을 불러와서 inference한 결과를 db에 새로운 열로서 추가
db_upload.py 실행후에 추가

데이터 행을 한줄한줄 불러오고 모델도 돌려야하기에 시간이 상당히 걸림

실행방법:
python add_inference.py

'''


def load_model():
    """
    모델과 토크나이저 로드
    """
    model_name_or_path = "../bubble_free_BERT"
    tokenizer_name_or_path = "../bubble_free_tokenizer"

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    
    return model, tokenizer

def predict(model, tokenizer, content):
    """
    모델을 사용하여 추론 수행
    """
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(content, return_tensors="pt", max_length=128, truncation=True, padding="max_length")

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        softmax = nn.Softmax(dim=1)  # dim=1은 클래스 차원에 대해 softmax를 적용
        preds = softmax(logits)

        


    preds_rounded = preds.cpu().numpy().tolist()
    logits_rounded = logits.cpu().numpy().tolist()


    return json.dumps(preds_rounded), json.dumps(logits_rounded)

def add_inference_columns(cursor):
    """
    데이터베이스에 새로운 열 'preds_rounded'와 'logits_rounded' 추가
    """
    try:
        cursor.execute(
            """
            ALTER TABLE article
            ADD COLUMN preds_rounded TEXT
            """
        )
        cursor.execute(
            """
            ALTER TABLE article
            ADD COLUMN logits_rounded TEXT
            """
        )
        print("새로운 열 'preds_rounded'와 'logits_rounded'가 추가되었습니다.")
    except Error as err:
        print(f"열 추가 오류: {err}")

def update_inference(cursor, article_id, preds_rounded, logits_rounded):
    """
    데이터베이스에 예측 결과 업데이트
    """
    try:
        cursor.execute(
            """
            UPDATE article
            SET preds_rounded = %s, logits_rounded = %s
            WHERE article_id = %s
            """,
            (preds_rounded, logits_rounded, article_id)
        )
        print(f"article_id {article_id}의 예측 결과가 업데이트되었습니다.")
    except Error as err:
        print(f"예측 결과 업데이트 오류: {err}")

# 데이터베이스 설정
config_path = "."
config_file = os.path.join(config_path, "db.ini")

# ConfigParser 객체 생성 및 설정 파일 읽기
config = configparser.ConfigParser()
config.read(config_file)

try:
    # MySQL 연결
    conn = mysql.connector.connect(
        host=config.get("DB", "host"),
        port=config.getint("DB", "port"),
        user=config.get("DB", "user"),
        password=config.get("DB", "password"),
        database=config.get("DB", "database")
    )
    if conn.is_connected():
        print("데이터베이스에 성공적으로 연결되었습니다.")
        cursor = conn.cursor()

        # 새로운 열 'preds_rounded'와 'logits_rounded' 추가
        add_inference_columns(cursor)
        
        # 모델 로드
        model, tokenizer = load_model()
        
        # 데이터베이스에서 데이터 불러오기
        cursor.execute("SELECT article_id, content FROM article")
        rows = cursor.fetchall()
        
        for row in rows:
            article_id, content = row
            preds_rounded, logits_rounded = predict(model, tokenizer, content)
            update_inference(cursor, article_id, preds_rounded, logits_rounded)
        
        conn.commit()

except Error as err:
    print(f"오류: {err}")
finally:
    if conn.is_connected():
        cursor.close()
        conn.close()
        print("연결이 종료되었습니다.")