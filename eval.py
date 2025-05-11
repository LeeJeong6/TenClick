from sentence_transformers import SentenceTransformer, util
import json

# 모델 불러오기
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# JSON 파일에서 데이터 불러오기
with open('/mnt/hdd_6tb/bill0914/oth/labels.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 각 인스턴스를 검색 문장으로 구성
def build_search_string(entry):
    t = entry["text"]
    parts = [
        t.get("색상", ""),
        t.get("소매기장", ""),
        t.get("기장", ""),
        t.get("카테고리", ""),
        t.get("스타일", ""),
        t.get("서브스타일", ""),
        t.get("넥라인", ""),
        t.get("핏", "")
    ]
    return ' '.join([p for p in parts if p])  # 빈 값 제외

corpus = [build_search_string(entry) for entry in data]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# 사용자 입력 쿼리
query = "블랙 원피스"
query_embedding = model.encode(query, convert_to_tensor=True)

# 코사인 유사도 계산
cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

# Top 100개 결과 인덱스와 유사도 추출
top_results = sorted(
    list(enumerate(cos_scores)),
    key=lambda x: x[1],
    reverse=True
)[:5]

# 결과 출력
print(f"쿼리: {query}")
print(f"Top 100 유사 결과:")
for idx, score in top_results:
    print(f"\n[유사도: {score.item():.4f}] 이미지: {data[idx]['image']}")
    print(json.dumps(data[idx]['text'], ensure_ascii=False, indent=2))