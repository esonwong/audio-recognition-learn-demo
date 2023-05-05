let recognizer;

const classMap = ["啊", "哦", "呀", "嗯", "Other"];

const rangeContiner = document.querySelector("#ranges");

const buttonContiner = document.querySelector("#buttons");

const result = document.querySelector("#result");

classMap.forEach((className, i) => {
  const button = document.createElement("button");
  button.innerText = className;
  button.setAttribute("onmousedown", `collect(${i})`);
  button.setAttribute("onmouseup", `collect(null)`);
  buttonContiner.appendChild(button);
});

classMap.forEach((className, i) => {
  const input = document.createElement("input");
  input.setAttribute("type", "range");
  input.setAttribute("id", `output-${i}`);
  input.setAttribute("class", "output");
  input.setAttribute("min", "0");
  input.setAttribute("max", "1");
  input.setAttribute("step", "0.01");
  input.value = 0;

  const span = document.createElement("span");
  span.innerText = 0;

  rangeContiner.appendChild(input);
  rangeContiner.appendChild(document.createTextNode(className));
  rangeContiner.appendChild(span);

  rangeContiner.appendChild(document.createElement("br"));
  rangeContiner.appendChild(document.createElement("br"));
});

const LOCAL_NAME = "indexeddb://my-model";

function predictWord() {
  // Array of words that the recognizer is trained to recognize.
  const words = recognizer.wordLabels();
  recognizer.listen(
    ({ scores }) => {
      // Turn scores into a list of (score,word) pairs.
      scores = Array.from(scores).map((s, i) => ({ score: s, word: words[i] }));
      // Find the most probable word.
      scores.sort((s1, s2) => s2.score - s1.score);
      document.querySelector("#console").textContent = scores[0].word;
    },
    { probabilityThreshold: 0.75 }
  );
}

async function app() {
  recognizer = speechCommands.create("BROWSER_FFT");
  document.querySelector("#console").textContent = "Loading...";
  await recognizer.ensureModelLoaded();
  document.querySelector("#console").textContent = "Loaded";
  // predictWord();
  loadModelFromLocalStorage();
}

app();

// One frame is ~23ms of audio.
const NUM_FRAMES = 9;
let examples = [];
const NUM_CLASSWS = classMap.length;

function collect(label) {
  if (recognizer.isListening()) {
    return recognizer.stopListening();
  }
  if (label == null) {
    return;
  }
  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      examples.push({ vals, label });
      document.querySelector(
        "#console"
      ).textContent = `${examples.length} examples collected`;
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    }
  );
}

function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map((x) => (x - mean) / std);
}

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
  buildModel();
  toggleButtons(false);
  const ys = tf.oneHot(
    examples.map((e) => e.label),
    NUM_CLASSWS
  );
  const xsShape = [examples.length, ...INPUT_SHAPE];
  const xs = tf.tensor(flatten(examples.map((e) => e.vals)), xsShape);

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector("#console").textContent = `Accuracy: ${(
          logs.acc * 100
        ).toFixed(1)}% Epoch: ${epoch + 1}`;
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleButtons(true);
}

async function loadModelFromLocalStorage() {
  try {
    model = await tf.loadLayersModel(LOCAL_NAME);
    console.log("模型已从本地存储加载:", model);
    return model;
  } catch (error) {
    console.error("加载模型时发生错误:", error);
    buildModel();
    return null;
  }
}

function buildModel() {
  model = tf.sequential();
  model.add(
    tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [NUM_FRAMES, 3],
      activation: "relu",
      inputShape: INPUT_SHAPE
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: NUM_CLASSWS, activation: "softmax" }));
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
}

function toggleButtons(enable) {
  document.querySelectorAll("button").forEach((b) => (b.disabled = !enable));
}

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}

async function showResult(labelTensor) {
  const data = await labelTensor.data();
  const probabilityThreshold = document.querySelector("#threshold").value;

  result.textContent = "---";

  data.forEach((prob, i) => {
    const input = document.querySelector(`#output-${i}`);
    input.value = prob.toFixed(2);
    input.nextElementSibling.textContent = prob.toFixed(2);

    if (prob > probabilityThreshold) {
      input.nextElementSibling.classList.add("active");
      result.textContent = classMap[i];
    } else {
      input.nextElementSibling.classList.remove("active");
    }
  });
}

function listen() {
  if (recognizer.isListening()) {
    recognizer.stopListening();
    toggleButtons(true);
    document.getElementById("listen").textContent = "Listen";
    return;
  }
  toggleButtons(false);
  document.getElementById("listen").textContent = "Stop";
  document.getElementById("listen").disabled = false;

  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
      const probs = model.predict(input);
      // const predLabel = probs.argMax(1);
      await showResult(probs);
      tf.dispose([input, probs]);
      // tf.dispose([input, probs, predLabel]);
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    }
  );
}

// 导出出模型
async function exportModel() {
  const saveResult = await model.save("downloads://my-model");
  console.log(saveResult);
}

async function saveModel() {
  await model.save(LOCAL_NAME);
}
