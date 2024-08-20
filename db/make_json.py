import pandas as pd
import os
import json

"""
db/make_json.py: raw_data에 article_data url 기준으로 inner join하여 json파일 생성
-> innerjoin이 안된 기사들 존재? why?

실행방법:
python make_json.py

향후 개선방안:
innerjoin이 안된 기사들 해결
"""

# 경로 설정
raw_data_path = "../src/raw_data"
article_data_path = "../src/article_data"
data_path = "../db/json"

# 파일 접두사 리스트
file_prefixes = ['chosun', 'donga', 'hani', 'joongang', 'khan']

# 매칭되지 않은 횟수를 저장할 딕셔너리
unmatched_counts = {}

# 각 파일에 대한 처리
for prefix in file_prefixes:
    # 파일 경로
    raw_csv = os.path.join(raw_data_path, f"{prefix}.csv")
    article_csv = os.path.join(article_data_path, f"{prefix}_article.csv")
    
    # CSV 파일 로드
    raw_df = pd.read_csv(raw_csv)
    article_df = pd.read_csv(article_csv)
    
    # 'donga' 파일의 경우 'Body' 열 이름을 'Article'로 변경
    if prefix == 'donga':
        article_df.rename(columns={'Body': 'Article'}, inplace=True)
    
    # URL을 기준으로 병합 (left join)
    merged_df = pd.merge(raw_df, article_df[['URL', 'Article']], on='URL', how='left')
    
    # NaN 값을 빈 문자열로 처리
    merged_df.fillna('', inplace=True)
    
    # 매칭되지 않은 행 계산 (Article 열이 빈 문자열인 경우)
    unmatched_count = (merged_df['Article'] == '').sum()
    unmatched_counts[prefix] = unmatched_count
    
    # 결과를 JSON으로 변환
    json_data = merged_df.to_dict(orient='records')
    
    # JSON 파일 저장
    json_file_path = os.path.join(data_path, f"{prefix}.json")
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, ensure_ascii=False, indent=4)

# 매칭되지 않은 횟수 출력
for prefix, count in unmatched_counts.items():
    print(f"{prefix} 신문사에서 매칭되지 않은 URL 개수: {count}")

print("작업 완료! JSON 파일이 저장되었으며 매칭되지 않은 URL 개수가 출력되었습니다.")

