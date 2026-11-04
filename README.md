# 손글씨 숫자 인식기 (Handwritten Digit Recognizer)

마우스나 손가락으로 숫자(0~9)를 그리면 MNIST로 학습한 신경망이 어떤 숫자인지 인식합니다.

**▶ 웹 데모: https://omilk339-debug.github.io/digit-recognizer/**

- 웹 버전은 **서버 없이 브라우저에서만** 동작합니다 (외부 라이브러리 없음, 순수 JavaScript)
- 데스크톱 버전(Python + Tkinter)도 함께 들어 있습니다
- 테스트 정확도 **98.1%**

## 동작 원리

```
그림 (280×280)
  → 블러 → 그린 영역 잘라내기 → 20×20으로 축소 → 28×28 중앙 배치 → 무게중심 정렬
  → 신경망 784 → 256 (ReLU) → 128 (ReLU) → 10 (softmax)
  → 0~9 각 숫자의 확률
```

MNIST 이미지와 같은 형태가 되도록 전처리한 뒤 신경망에 넣습니다.
웹 화면에서 "Model input (28×28)"으로 모델이 실제로 보는 이미지를 확인할 수 있습니다.

## 프로젝트 구조

| 파일 | 설명 |
|---|---|
| `digit_recognizer.py` | 모델 학습 + 데스크톱 앱 (Tkinter) |
| `export_weights.py` | 학습된 가중치를 `docs/model.json`으로 내보내기 |
| `docs/index.html`, `docs/style.css` | 웹 화면 |
| `docs/recognizer.js` | 전처리 + 신경망 계산 (Python 버전을 JavaScript로 구현) |
| `docs/app.js` | 그림판 그리기와 결과 표시 |
| `docs/model.json` | 학습된 가중치 (약 1.8MB) |

## 실행 방법

### 웹 버전 (로컬)

```bash
cd docs
python -m http.server 8000
```

브라우저에서 http://localhost:8000 을 엽니다.
(`index.html`을 더블클릭해서 열면 보안 정책 때문에 `model.json`을 불러오지 못합니다.)

### 데스크톱 버전

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python digit_recognizer.py
```

처음 실행할 때 MNIST 데이터를 내려받고 모델을 학습합니다 (1~2분).

### 모델을 다시 학습한 경우

```bash
python export_weights.py
```

`docs/model.json`이 새 가중치로 바뀌면 웹 버전에도 반영됩니다.

## GitHub Pages 배포

저장소 **Settings → Pages → Build and deployment**에서
Source를 `Deploy from a branch`, 브랜치를 `main`, 폴더를 `/docs`로 지정합니다.

## 사용 기술

Python · scikit-learn (MLPClassifier) · NumPy · Pillow · Tkinter · HTML/CSS/JavaScript (Canvas API)
