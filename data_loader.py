"""이미지만 불러와서 npy로 벡터저장소 만드는 파일"""
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from fashion_clip.fashion_clip import FashionCLIP

# 초기화
fclip = FashionCLIP('fashion-clip')
image_dir = "/mnt/hdd_6tb/bill0914/oth/clothes_dataset"  # 압축 해제된 이미지 폴더 경로
output_path = "/mnt/hdd_6tb/bill0914/oth/result/image_embeddings.npy"

image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(('.jpg'))
])

# 이미지 로딩 및 벡터화 (batch 처리)
batch_size = 32
all_embeddings = []

for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_paths = image_paths[i:i+batch_size]
    batch_images = [Image.open(p).convert("RGB") for p in batch_paths]

    # 인코딩
    embeddings = fclip.encode_images(batch_images, batch_size=len(batch_images))
    
    # L2 정규화
    embeddings /= np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

    all_embeddings.append(embeddings)

# 전체 벡터 하나로 합치기
final_embeddings = np.vstack(all_embeddings)

# npy 저장
np.save(output_path, final_embeddings)
print(f"✅ 저장 완료: {output_path} (shape: {final_embeddings.shape})")
