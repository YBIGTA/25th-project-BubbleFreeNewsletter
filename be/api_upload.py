from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, or_, and_
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import configparser
import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
import json
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Optional
from scipy.spatial.distance import cosine


'''

uvicorn api_upload:app --reload
'''

# FastAPI 애플리케이션 설정
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프론트엔드 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터베이스 설정
config = configparser.ConfigParser()
config_path = "../db/db.ini"  # 데이터베이스 설정 파일 경로
config.read(config_path)

DATABASE_URL = f"mysql+mysqlconnector://{config['DB']['user']}:{config['DB']['password']}@{config['DB']['host']}:{config['DB']['port']}/{config['DB']['database']}"

# SQLAlchemy 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Article 모델 정의
class Article(Base):
    __tablename__ = "article"
    
    article_id = Column(String(255), primary_key=True, index=True)
    title = Column(Text, index=True)
    keyword = Column(Text)
    content = Column(Text)
    date = Column(Integer)
    logits_rounded = Column(JSON)  # BERT 모델 추론 후 logits 저장

# UserInfo 모델 정의
class UserInfo(Base):
    __tablename__ = "user_info"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    average_logits = Column(Text)  # JSON 문자열로 저장

# 테이블 생성 함수
def init_db():
    Base.metadata.create_all(bind=engine)

# 비동기 데이터베이스 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# BERT 모델 및 토크나이저 로드
model = BertForSequenceClassification.from_pretrained("../bubble_free_BERT")
tokenizer = BertTokenizer.from_pretrained("../bubble_free_tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI application!"}

@app.get("/search/{query}")
async def search_by_query(query: str, db: Session = Depends(get_db)):
    results = db.query(Article).filter(
        or_(
            Article.title.ilike(f"%{query}%"),
            Article.keyword.ilike(f"%{query}%")
        )
    ).all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No articles found")
    
    return results

@app.post("/article/infer/{article_id}")
async def infer_article(article_id: str, db: Session = Depends(get_db)):
    article = db.query(Article).filter(Article.article_id == article_id).first()

    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    content = article.content
    inputs = tokenizer(content, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        softmax = nn.Softmax(dim=1)
        preds = softmax(logits)

    preds_rounded = preds.cpu().numpy().tolist()
    logits_rounded = logits.cpu().numpy().tolist()

    preds_json = json.dumps(preds_rounded)
    logits_json = json.dumps(logits_rounded)

    article.inference = {"softmax_probabilities": preds_json, "logits": logits_json}
    db.commit()

    return {"softmax_probabilities": preds_rounded, "logits": logits_rounded}

@app.get("/articles/latest")
async def get_latest_articles(db: Session = Depends(get_db)):
    articles = db.query(Article).order_by(Article.date.desc()).limit(20).all()
    return articles

class Selection(BaseModel):
    article_id: str
    selected: bool
    logits: list

class SelectionsRequest(BaseModel):
    user_id: str
    selections: list[Selection]

@app.post("/save-selections")
async def save_selections(request: SelectionsRequest, db: Session = Depends(get_db)):
    user_id = request.user_id
    selections = request.selections
    
    logits_sum = None
    count = 0

    for selection in selections:
        if selection.selected:
            logits = selection.logits
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum = [x + y for x, y in zip(logits_sum, logits)]
            count += 1
    
    if count == 0:
        raise HTTPException(status_code=400, detail="No valid selections provided.")

    average_logits = [x / count for x in logits_sum]
    average_logits_json = json.dumps(average_logits)

    user_info = db.query(UserInfo).filter(UserInfo.user_id == user_id).first()
    if user_info:
        user_info.average_logits = average_logits_json
    else:
        user_info = UserInfo(user_id=user_id, average_logits=average_logits_json)
        db.add(user_info)

    db.commit()

    return {"message": "Selections saved successfully!", "average_logits": average_logits}




@app.get("/search/")
async def search_articles(user_id: str, db: Session = Depends(get_db), date: Optional[int] = None, query: Optional[str] = None):
    # 날짜 검증
    try:
        if date:
            date_str = str(date)
            datetime.strptime(date_str, "%Y%m%d")
            if datetime.strptime(date_str, "%Y%m%d") > datetime.now():
                raise HTTPException(status_code=400, detail="입력한 날짜가 오늘 날짜를 초과했습니다.")
        else:
            date = int((datetime.now() - timedelta(days=3)).strftime("%Y%m%d"))
    except ValueError:
        raise HTTPException(status_code=400, detail="날짜 형식이 맞지 않습니다. YYYYMMDD 형식으로 입력해 주세요.")

    start_date = date - 7  # 앞뒤로 일주일 범위
    end_date = date + 7

    # query 조건 설정
    query_conditions = []
    if query:
        query_conditions.append(
            or_(
                Article.title.ilike(f"%{query}%"),
                Article.keyword.ilike(f"%{query}%")
            )
        )
    query_conditions.append(and_(Article.date >= start_date, Article.date <= end_date))

    articles = db.query(Article).filter(*query_conditions).all()

    if not articles:
        raise HTTPException(status_code=404, detail="조건에 맞는 기사가 없습니다.")

    # 유저의 average_logits 가져오기
    user_info = db.query(UserInfo).filter(UserInfo.user_id == user_id).first()
    if not user_info or not user_info.average_logits:
        raise HTTPException(status_code=404, detail="해당 유저의 평균 로그정보가 없습니다.")

    average_logits = json.loads(user_info.average_logits)

    # 각 기사와 유저 average_logits의 코사인 유사도 계산
    cosine_similarities = []
    for article in articles:
        logits_rounded = json.loads(article.logits_rounded)
        similarity = 1 - cosine(average_logits, logits_rounded)
        cosine_similarities.append((article, similarity))

    # 가장 가까운 기사와 먼 기사 찾기
    closest_article = max(cosine_similarities, key=lambda x: x[1])[0]
    furthest_article = min(cosine_similarities, key=lambda x: x[1])[0]

    return {"closest_article": closest_article, "furthest_article": furthest_article}

# DB 초기화
init_db()
