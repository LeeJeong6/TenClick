# Fashion CLIP 기반 이미지 검색 프로젝트

이 프로젝트는 [Fashion CLIP](https://github.com/patrickjohncyh/fashion-clip) 모델을 활용하여 패션 이미지를 텍스트 설명과 매칭하는 시스템입니다. 모델은 Hugging Face API를 통해 로드됩니다.

## ✅ 프로젝트 개요

- **모델**: Fashion CLIP
- **데이터**: AIHUB의 K-FASHION 이미지 데이터셋 (약 13,000장)
- **라벨 포맷**: 각 이미지에 대해 다음과 같은 형식으로 JSON으로 전처리
- **앞으로의 계획**: prompt에 입력되게 해서 웹사이트로 발전시킬 예정입니다.

```json
{
  "image": "1190511.jpg",
  "text": {
    "스타일": "스트리트",
    "서브스타일": "",
    "카테고리": "티셔츠",
    "기장": "크롭",
    "색상": "화이트",
    "소매기장": "긴팔",
    "넥라인": "라운드넥",
    "핏": "노멀"
  }
}





디렉토리 구조는 아래와 같습니다
.
|
├── data/
|   └──clothes_dataset/
|       └── *.jpg   
|   └──labels.json
├── data_loader.py         
├── inference.py           
├── result/
│   └── image_embeddings.npy
|   └── top9_grid.jpg
├── decrease_size
|   └── 개발중,,,
│   
         
