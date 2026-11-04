"""
손글씨 숫자 인식기.

마우스로 캔버스에 숫자(0~9)를 그리면 프로그램이 어떤 숫자인지 예측합니다.
처음 실행할 때 MNIST 데이터셋으로 신경망(MLP)을 학습하고,
학습된 모델을 디스크에 저장해 두어 다음 실행부터는 바로 시작됩니다.
"""

import os
import urllib.request

import joblib
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "mnist.npz")
MODEL_PATH = os.path.join(BASE_DIR, "data", "mnist_mlp.joblib")
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"

CANVAS_SIZE = 280  # 그림판 크기(픽셀) - MNIST 28x28의 10배
BRUSH_RADIUS = 10  # 붓 반지름(픽셀)


def load_mnist():
    """MNIST 데이터가 없으면 내려받고 (x_train, y_train, x_test, y_test)를 반환합니다."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(MNIST_URL, DATA_PATH)
    with np.load(DATA_PATH) as data:
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def train_model():
    """MNIST로 MLP 분류기를 학습하고 디스크에 저장합니다."""
    x_train, y_train, x_test, y_test = load_mnist()
    # 28x28 이미지를 784 길이의 벡터로 펼치고 0~1 범위로 정규화
    x_train = x_train.reshape(len(x_train), -1).astype(np.float32) / 255.0
    x_test = x_test.reshape(len(x_test), -1).astype(np.float32) / 255.0

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),  # 은닉층 2개 (뉴런 256개, 128개)
        max_iter=30,
        early_stopping=True,  # 검증 점수가 더 이상 오르지 않으면 학습 중단
        random_state=42,
        verbose=True,
    )
    print("Training model (this takes a minute or two)...")
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")

    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train_model():
    """저장된 모델이 있으면 불러오고, 없으면 새로 학습합니다."""
    if os.path.exists(MODEL_PATH):
        print("Loading cached model...")
        return joblib.load(MODEL_PATH)
    return train_model()


def preprocess(image):
    """
    캔버스 이미지를 MNIST 형식의 784 길이 벡터로 변환합니다.

    MNIST 숫자는 검은 배경에 흰 글씨이고, 20x20 영역 안에 들어가도록 크기가 맞춰져 있으며,
    28x28 프레임 안에서 무게중심이 가운데에 오도록 배치되어 있습니다. 여기서 이를 똑같이 재현합니다.
    """
    pixels = np.array(image, dtype=np.float32)
    rows = np.any(pixels > 0, axis=1)
    cols = np.any(pixels > 0, axis=0)
    if not rows.any():
        return None  # 아무것도 그리지 않음

    # 그린 영역만 잘라내기 (경계 상자)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    digit = image.crop((left, top, right + 1, bottom + 1))

    # 가로세로 비율을 유지하면서 긴 쪽이 20픽셀이 되도록 축소
    width, height = digit.size
    scale = 20.0 / max(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    digit = digit.resize(new_size, Image.LANCZOS)

    # 28x28 프레임에 붙인 뒤, 무게중심이 정중앙에 오도록 이동
    frame = Image.new("L", (28, 28), 0)
    frame.paste(digit, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    arr = np.array(frame, dtype=np.float32)
    ys, xs = np.nonzero(arr)
    weights = arr[ys, xs]
    shift_y = int(round(14 - np.average(ys, weights=weights)))
    shift_x = int(round(14 - np.average(xs, weights=weights)))
    arr = np.roll(arr, (shift_y, shift_x), axis=(0, 1))

    return (arr / 255.0).reshape(1, -1)


class DigitApp:
    """그림판과 예측 결과를 보여주는 Tkinter GUI."""

    def __init__(self, root, model):
        self.model = model
        self.root = root
        root.title("Handwritten Digit Recognizer")
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="cross")
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)

        # 전처리를 위해 화면 캔버스와 똑같이 그려지는 보이지 않는 PIL 이미지
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_point = None

        self.result_label = tk.Label(root, text="Draw a digit", font=("Arial", 24, "bold"),
                                     width=12)
        self.result_label.grid(row=0, column=1, padx=10)

        self.prob_label = tk.Label(root, text="", font=("Consolas", 12), justify="left")
        self.prob_label.grid(row=1, column=1, padx=10, sticky="n")

        buttons = tk.Frame(root)
        buttons.grid(row=2, column=1, pady=10)
        tk.Button(buttons, text="Recognize", width=10, command=self.predict).pack(side="left", padx=5)
        tk.Button(buttons, text="Clear", width=10, command=self.clear).pack(side="left", padx=5)

        # 마우스 드래그로 그리기, 버튼을 떼면 자동 인식
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        # 단축키: Enter = 인식, Esc = 지우기
        root.bind("<Return>", lambda _: self.predict())
        root.bind("<Escape>", lambda _: self.clear())

    def on_drag(self, event):
        """마우스를 따라 굵은 선을 그립니다."""
        x, y = event.x, event.y
        if self.last_point is None:
            self.last_point = (x, y)
        x0, y0 = self.last_point
        self.canvas.create_line(x0, y0, x, y, fill="white", width=BRUSH_RADIUS * 2,
                                capstyle=tk.ROUND, smooth=True)
        self.draw.line([x0, y0, x, y], fill=255, width=BRUSH_RADIUS * 2)
        self.draw.ellipse([x - BRUSH_RADIUS, y - BRUSH_RADIUS,
                           x + BRUSH_RADIUS, y + BRUSH_RADIUS], fill=255)
        self.last_point = (x, y)

    def on_release(self, _event):
        """획을 마치고 자동으로 예측합니다."""
        self.last_point = None
        self.predict()

    def predict(self):
        """현재 그림을 모델에 넣어 결과를 표시합니다."""
        # 살짝 흐리게 처리해 MNIST의 부드러운 가장자리와 비슷하게 만듦
        features = preprocess(self.image.filter(ImageFilter.GaussianBlur(1)))
        if features is None:
            self.result_label.config(text="Draw a digit")
            self.prob_label.config(text="")
            return
        probs = self.model.predict_proba(features)[0]
        best = int(np.argmax(probs))
        self.result_label.config(text=f"Result: {best}  ({probs[best]:.0%})")

        # 확률이 높은 상위 3개 후보 표시
        top3 = np.argsort(probs)[::-1][:3]
        self.prob_label.config(text="\n".join(f"{d}: {probs[d]:6.1%}" for d in top3))

    def clear(self):
        """캔버스와 내부 이미지를 초기화합니다."""
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
        self.result_label.config(text="Draw a digit")
        self.prob_label.config(text="")


def main():
    model = load_or_train_model()
    root = tk.Tk()
    DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
