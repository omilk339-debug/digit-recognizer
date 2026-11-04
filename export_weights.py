"""
학습된 MLP 모델의 가중치를 웹 버전에서 쓸 수 있도록 JSON으로 내보냅니다.

사용법:
    python export_weights.py

결과 파일 docs/model.json 은 브라우저(docs/recognizer.js)에서 불러와
신경망 계산(순전파)에 사용됩니다.
"""

import json
import os

from digit_recognizer import BASE_DIR, load_or_train_model

OUTPUT_PATH = os.path.join(BASE_DIR, "docs", "model.json")
DECIMALS = 5  # 소수점 자릿수 - 파일 크기를 줄이면서 정확도는 유지


def main():
    model = load_or_train_model()

    layers = []
    for weights, biases in zip(model.coefs_, model.intercepts_):
        layers.append({
            "inputs": int(weights.shape[0]),
            "outputs": int(weights.shape[1]),
            # 가중치 행렬을 행 우선(row-major) 1차원 배열로 펼침: index = i * outputs + j
            "weights": [round(float(v), DECIMALS) for v in weights.ravel()],
            "biases": [round(float(v), DECIMALS) for v in biases],
        })

    data = {
        "hiddenActivation": model.activation,      # 은닉층 활성화 함수 (relu)
        "outputActivation": model.out_activation_,  # 출력층 활성화 함수 (softmax)
        "layers": layers,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
