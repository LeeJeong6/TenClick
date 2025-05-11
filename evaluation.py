import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 모델과 프로세서 로드
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# 비교할 텍스트
texts = ["blue onepiece with white shirt"]

# 이미지가 저장된 폴더
image_dir = "/mnt/hdd_6tb/bill0914/oth/clothes_dataset"
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

# npy 파일 로드 (이미지 벡터)
image_embeddings = np.load("/mnt/hdd_6tb/bill0914/oth/result/image_embeddings_100_imagesssssssssss.npy")

# 텍스트 임베딩
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
text_outputs = model.get_text_features(**text_inputs)
text_embeddings = text_outputs.detach().cpu().numpy()

# 유사도 계산을 위한 L2 정규화
text_embeddings /= np.linalg.norm(text_embeddings, ord=2, axis=1, keepdims=True)
image_embeddings /= np.linalg.norm(image_embeddings, ord=2, axis=1, keepdims=True)

# 유사도 계산
similarities = np.dot(image_embeddings, text_embeddings.T)

# 가장 유사한 이미지 순위대로 5개 추출
top_5_indices = similarities.argmax(axis=1)  # 가장 유사한 텍스트에 대한 인덱스
top_5_similarities = np.max(similarities, axis=1)  # 유사도 순위

# 상위 5개의 이미지 추출
top_5_indices_sorted = np.argsort(top_5_similarities)[::-1][:5]  # 가장 유사한 이미지 5개 인덱스

# 저장할 경로
output_dir = "/mnt/hdd_6tb/bill0914/oth/result"
os.makedirs(output_dir, exist_ok=True)

# 상위 5개 이미지를 저장
for i, idx in enumerate(top_5_indices_sorted):
    best_image_path = os.path.join(image_dir, image_filenames[idx])
    best_image = Image.open(best_image_path).convert("RGB")
    
    # 이미지 저장
    output_image_path = os.path.join(output_dir, f"most_similar_{i+1}.jpg")
    best_image.save(output_image_path)
    print(f"가장 유사한 이미지 저장 완료: {output_image_path} (유사도: {top_5_similarities[idx]:.4f})")
