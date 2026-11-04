/*
 * 웹 화면 동작: 그림판 그리기, 인식 실행, 결과 표시.
 */
(function () {
  "use strict";

  const { preprocess, predict, FRAME } = window.DigitRecognizer;

  const canvas = document.getElementById("pad");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const preview = document.getElementById("preview");
  const previewCtx = preview.getContext("2d");
  const resultEl = document.getElementById("result");
  const confidenceEl = document.getElementById("confidence");
  const barsEl = document.getElementById("bars");
  const statusEl = document.getElementById("status");
  const clearBtn = document.getElementById("clear");

  const BRUSH_WIDTH = 20; // 붓 굵기 (데스크톱 버전과 동일)

  let model = null;
  let drawing = false;
  let lastPoint = null;

  // 0~9 확률 막대 그래프 영역 만들기
  const bars = [];
  for (let d = 0; d < 10; d++) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      `<span class="bar-label">${d}</span>` +
      `<span class="bar-track"><span class="bar-fill"></span></span>` +
      `<span class="bar-value">0%</span>`;
    barsEl.appendChild(row);
    bars.push({
      row,
      fill: row.querySelector(".bar-fill"),
      value: row.querySelector(".bar-value"),
    });
  }

  // 캔버스를 검은색으로 초기화
  function resetCanvas() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    previewCtx.clearRect(0, 0, preview.width, preview.height);
    resultEl.textContent = "?";
    confidenceEl.textContent = "Draw a digit";
    bars.forEach((b) => {
      b.fill.style.width = "0%";
      b.value.textContent = "0%";
      b.row.classList.remove("best");
    });
  }

  // 화면에 표시된 크기와 캔버스 실제 크기가 다를 수 있으므로 좌표를 환산
  function toCanvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) * canvas.width) / rect.width,
      y: ((event.clientY - rect.top) * canvas.height) / rect.height,
    };
  }

  function drawTo(point) {
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = BRUSH_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint = point;
  }

  // 마우스와 터치를 모두 지원하는 포인터 이벤트
  canvas.addEventListener("pointerdown", (event) => {
    drawing = true;
    canvas.setPointerCapture(event.pointerId);
    lastPoint = toCanvasPoint(event);
    drawTo(lastPoint); // 클릭만 해도 점이 찍히도록
  });

  canvas.addEventListener("pointermove", (event) => {
    if (drawing) drawTo(toCanvasPoint(event));
  });

  function endStroke() {
    if (!drawing) return;
    drawing = false;
    lastPoint = null;
    recognize();
  }
  canvas.addEventListener("pointerup", endStroke);
  canvas.addEventListener("pointercancel", endStroke);

  clearBtn.addEventListener("click", resetCanvas);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") resetCanvas();
  });

  // 캔버스 픽셀을 흑백 배열로 읽어 인식 실행
  function recognize() {
    if (!model) return;
    const { data, width, height } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const gray = new Float32Array(width * height);
    for (let i = 0; i < gray.length; i++) gray[i] = data[i * 4]; // 빨강 채널 = 밝기

    const input = preprocess(gray, width, height);
    if (!input) return;

    drawPreview(input);
    showResult(predict(model, input));
  }

  // 모델에 실제로 들어가는 28x28 이미지를 확대해서 보여줌
  function drawPreview(input) {
    const image = previewCtx.createImageData(FRAME, FRAME);
    for (let i = 0; i < input.length; i++) {
      const v = Math.round(input[i] * 255);
      image.data.set([v, v, v, 255], i * 4);
    }
    previewCtx.putImageData(image, 0, 0);
  }

  function showResult(probs) {
    const best = probs.indexOf(Math.max(...probs));
    resultEl.textContent = best;
    confidenceEl.textContent = `Confidence ${(probs[best] * 100).toFixed(1)}%`;
    probs.forEach((p, d) => {
      const pct = (p * 100).toFixed(1) + "%";
      bars[d].fill.style.width = pct;
      bars[d].value.textContent = pct;
      bars[d].row.classList.toggle("best", d === best);
    });
  }

  // 모델 가중치 불러오기
  resetCanvas();
  fetch("model.json")
    .then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then((data) => {
      model = data;
      statusEl.textContent = "Model loaded. Draw a digit (0-9) in the box.";
    })
    .catch((err) => {
      statusEl.textContent = `Failed to load model: ${err.message}`;
      statusEl.classList.add("error");
    });
})();
