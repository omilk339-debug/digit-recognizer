/*
 * 손글씨 숫자 인식 핵심 로직 (전처리 + 신경망 순전파).
 * Python 버전(digit_recognizer.py)의 preprocess()와 MLP 계산을 그대로 옮겼습니다.
 * 외부 라이브러리 없이 순수 JavaScript로 동작합니다.
 */
(function (global) {
  "use strict";

  const FRAME = 28; // MNIST 이미지 크기 (28x28)
  const BOX = 20;   // 숫자가 들어갈 영역 크기 (20x20)

  // 분리형 가우시안 블러 (PIL의 GaussianBlur(1)에 해당)
  function gaussianBlur(src, width, height, sigma) {
    const radius = Math.ceil(sigma * 3);
    const kernel = [];
    let sum = 0;
    for (let i = -radius; i <= radius; i++) {
      const v = Math.exp(-(i * i) / (2 * sigma * sigma));
      kernel.push(v);
      sum += v;
    }
    for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

    // 가로 방향 → 세로 방향 순서로 두 번 적용
    const tmp = new Float32Array(width * height);
    const out = new Float32Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          const xx = Math.min(width - 1, Math.max(0, x + k));
          acc += src[y * width + xx] * kernel[k + radius];
        }
        tmp[y * width + x] = acc;
      }
    }
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let acc = 0;
        for (let k = -radius; k <= radius; k++) {
          const yy = Math.min(height - 1, Math.max(0, y + k));
          acc += tmp[yy * width + x] * kernel[k + radius];
        }
        out[y * width + x] = acc;
      }
    }
    return out;
  }

  // 1차원 면적 평균 리샘플링: 원본 n개 → 대상 m개 (축소 시 계단 현상 방지)
  function resample1D(get, n, m) {
    const out = new Float32Array(m);
    const scale = n / m;
    for (let j = 0; j < m; j++) {
      const start = j * scale;
      const end = start + scale;
      let acc = 0;
      for (let i = Math.floor(start); i < Math.ceil(end); i++) {
        const overlap = Math.min(end, i + 1) - Math.max(start, i);
        if (overlap > 0) acc += get(Math.min(i, n - 1)) * overlap;
      }
      out[j] = acc / scale;
    }
    return out;
  }

  // 2차원 면적 평균 리샘플링 (가로 → 세로)
  function resize(src, sw, sh, dw, dh) {
    const rows = new Float32Array(dw * sh);
    for (let y = 0; y < sh; y++) {
      const row = resample1D((x) => src[y * sw + x], sw, dw);
      rows.set(row, y * dw);
    }
    const out = new Float32Array(dw * dh);
    for (let x = 0; x < dw; x++) {
      const col = resample1D((y) => rows[y * dw + x], sh, dh);
      for (let y = 0; y < dh; y++) out[y * dw + x] = col[y];
    }
    return out;
  }

  /*
   * 캔버스의 흑백 픽셀(0~255, 검은 배경에 흰 글씨)을 MNIST 형식의 784 길이 벡터로 변환합니다.
   * 1) 블러  2) 그린 부분만 잘라내기  3) 긴 쪽을 20픽셀로 축소
   * 4) 28x28 중앙에 배치  5) 무게중심을 정중앙으로 이동  6) 0~1로 정규화
   * 아무것도 그리지 않았으면 null을 반환합니다.
   */
  function preprocess(gray, width, height) {
    const blurred = gaussianBlur(gray, width, height, 1);

    // 경계 상자 찾기 (0.5 미만은 PIL처럼 정수 반올림 시 0이 되므로 무시)
    let top = height, bottom = -1, left = width, right = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (blurred[y * width + x] >= 0.5) {
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }
    if (bottom < 0) return null;

    // 잘라내기
    const cw = right - left + 1;
    const ch = bottom - top + 1;
    const cropped = new Float32Array(cw * ch);
    for (let y = 0; y < ch; y++) {
      for (let x = 0; x < cw; x++) {
        cropped[y * cw + x] = blurred[(y + top) * width + (x + left)];
      }
    }

    // 비율을 유지하며 긴 쪽이 20픽셀이 되도록 축소
    const scale = BOX / Math.max(cw, ch);
    const nw = Math.max(1, Math.round(cw * scale));
    const nh = Math.max(1, Math.round(ch * scale));
    const small = resize(cropped, cw, ch, nw, nh);

    // 28x28 프레임 중앙에 붙이기
    const frame = new Float32Array(FRAME * FRAME);
    const ox = Math.floor((FRAME - nw) / 2);
    const oy = Math.floor((FRAME - nh) / 2);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        frame[(y + oy) * FRAME + (x + ox)] = small[y * nw + x];
      }
    }

    // 무게중심 계산 후 정중앙(14, 14)으로 이동
    let total = 0, sumX = 0, sumY = 0;
    for (let y = 0; y < FRAME; y++) {
      for (let x = 0; x < FRAME; x++) {
        const v = frame[y * FRAME + x];
        total += v;
        sumX += v * x;
        sumY += v * y;
      }
    }
    const dx = Math.round(FRAME / 2 - sumX / total);
    const dy = Math.round(FRAME / 2 - sumY / total);
    const shifted = new Float32Array(FRAME * FRAME);
    for (let y = 0; y < FRAME; y++) {
      for (let x = 0; x < FRAME; x++) {
        const sx = x - dx;
        const sy = y - dy;
        if (sx >= 0 && sx < FRAME && sy >= 0 && sy < FRAME) {
          shifted[y * FRAME + x] = frame[sy * FRAME + sx] / 255;
        }
      }
    }
    return shifted;
  }

  // 신경망 순전파: 은닉층은 ReLU, 출력층은 softmax로 각 숫자의 확률을 계산
  function predict(model, input) {
    let x = input;
    model.layers.forEach((layer, index) => {
      const out = Float32Array.from(layer.biases);
      for (let i = 0; i < layer.inputs; i++) {
        const xi = x[i];
        if (xi === 0) continue; // 0인 입력은 결과에 영향이 없으므로 건너뜀 (속도 향상)
        const rowStart = i * layer.outputs;
        for (let j = 0; j < layer.outputs; j++) {
          out[j] += xi * layer.weights[rowStart + j];
        }
      }
      const isLast = index === model.layers.length - 1;
      if (!isLast) {
        for (let j = 0; j < out.length; j++) out[j] = Math.max(0, out[j]);
      }
      x = out;
    });

    // softmax (오버플로 방지를 위해 최댓값을 빼고 계산)
    const max = Math.max(...x);
    const exps = Array.from(x, (v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  global.DigitRecognizer = { preprocess, predict, FRAME };
})(typeof window !== "undefined" ? window : globalThis);
