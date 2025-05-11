"""image와 text embedding후 유사도 계산하는 파일"""
from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel

# 모델과 프로세서 로드
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# 비교할 텍스트
texts = ["blue shirt", "yellow onepiece"]

# 이미지가 저장된 폴더
image_dir = "/mnt/hdd_6tb/bill0914/oth/clothes_dataset"
image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

# 최고 확률 및 이미지 초기화
max_prob = -1
best_image = None
best_image_path = ""

# 이미지 반복
for filename in image_filenames:
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path).convert("RGB")

    # 전처리 및 모델 예측
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)  # shape: [1, len(texts)]

    # max_prob 기준으로 갱신
    top_prob = torch.max(probs).item()
    if top_prob > max_prob:
        max_prob = top_prob
        best_image = image
        best_image_path = image_path

# 가장 유사한 이미지 저장
if best_image is not None:
    output_path = "/mnt/hdd_6tb/bill0914/oth/result/most_similar.jpg"
    best_image.save(output_path)
    print(f"가장 유사한 이미지 저장 완료: {output_path} (확률: {max_prob:.4f})")
else:
    print("이미지를 찾을 수 없습니다.")
