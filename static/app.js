const $ = (id) => document.getElementById(id);
const presets = JSON.parse($("preset-data").textContent);
const styleFields = [
  ["anime-strength", "anime_strength"],
  ["line-detail", "line_detail"],
  ["color-vividness", "color_vividness"],
  ["background-mood", "background_mood"],
  ["photoreal-avoidance", "photoreal_avoidance"],
];

let currentImage = "";
let currentMetadata = {};
let rootPrompt = "";
let editInstructions = [];

function setError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function syncSlider(id) {
  $(id + "-value").value = $(id).value;
}

function applyPreset() {
  const preset = presets[$("preset").value];
  $("width").value = preset.width;
  $("height").value = preset.height;
  $("steps").value = preset.steps;
  $("cfg").value = preset.cfg;
  $("sampler").value = preset.sampler;
  $("clip-skip").value = preset.clip_skip;
  $("negative-prompt").value = preset.negative_prompt;
  styleFields.forEach(([id, key]) => {
    $(id).value = preset.style[key];
    syncSlider(id);
  });
}

function styleSettings() {
  return Object.fromEntries(styleFields.map(([id, key]) => [key, Number($(id).value)]));
}

function generationSettings() {
  return {
    preset: $("preset").value,
    refine_enabled: $("refine-enabled").checked,
    width: Number($("width").value),
    height: Number($("height").value),
    steps: Number($("steps").value),
    cfg: Number($("cfg").value),
    sampler: $("sampler").value,
    clip_skip: Number($("clip-skip").value),
    seed: Number($("seed").value),
    negative_prompt: $("negative-prompt").value,
    style_settings: styleSettings(),
  };
}

function fileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function beginProgress(message) {
  setError();
  $("empty").hidden = true;
  $("result").hidden = true;
  $("progress").hidden = false;
  $("progress-title").textContent = "準備中";
  $("progress-message").textContent = message;
  $("progress-fill").style.width = "0%";
}

function finishProgress() {
  $("progress").hidden = true;
  $("result").hidden = false;
}

async function runJob(payload) {
  const response = await fetch("/api/generate/start", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const start = await response.json();
  if (!response.ok) throw new Error(start.error || "処理を開始できませんでした");

  return new Promise((resolve, reject) => {
    const stream = new EventSource(`/api/generate/stream/${start.job_id}`);
    stream.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "status") {
        $("progress-title").textContent = data.phase === "generate" ? "画像を生成中" : "プロンプトを準備中";
        $("progress-message").textContent = data.message || "";
      } else if (data.type === "optimized_prompt") {
        $("progress-title").textContent = "画像を生成中";
        $("progress-message").textContent = "モデルへプロンプトを送信しました";
      } else if (data.type === "progress") {
        const percent = Math.round((data.step / data.total) * 100);
        $("progress-title").textContent = `画像を生成中 ${percent}%`;
        $("progress-message").textContent = `${data.step} / ${data.total} steps`;
        $("progress-fill").style.width = percent + "%";
      } else if (data.type === "done") {
        stream.close();
        resolve(data);
      } else if (data.type === "error") {
        stream.close();
        reject(new Error(data.message));
      }
    };
    stream.onerror = () => {
      stream.close();
      reject(new Error("サーバーとの接続が切れました"));
    };
  });
}

function showResult(data) {
  currentImage = data.image;
  currentMetadata = {
    original_prompt: rootPrompt || data.original_prompt,
    edit_history: [...editInstructions],
    optimized_prompt: data.optimized_prompt,
    optimizer_source: data.optimizer_source,
    intent_notes: data.intent_notes,
    refine_enabled: data.refine_enabled,
    generation_settings: data.settings,
  };
  $("result-image").src = "data:image/png;base64," + data.image;
  $("optimized-prompt").textContent = data.optimized_prompt;
  $("save-r2").textContent = "R2に保存";
  finishProgress();
}

$("preset").addEventListener("change", applyPreset);
styleFields.forEach(([id]) => $(id).addEventListener("input", () => syncSlider(id)));
applyPreset();

$("generate").addEventListener("click", async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) return setError("プロンプトを入力してください。");
  const button = $("generate");
  button.disabled = true;
  beginProgress("生成設定を確認しています");
  try {
    const data = await runJob({mode: "t2i", prompt, ...generationSettings()});
    rootPrompt = prompt;
    editInstructions = [];
    showResult(data);
    $("edit-history").innerHTML = "";
  } catch (error) {
    setError(error.message);
    $("progress").hidden = true;
    $("empty").hidden = false;
  } finally {
    button.disabled = false;
  }
});

$("edit").addEventListener("click", async () => {
  const instruction = $("edit-prompt").value.trim();
  if (!currentImage) return setError("編集元の画像がありません。");
  if (!instruction) return setError("修正内容を入力してください。");
  const button = $("edit");
  button.disabled = true;
  beginProgress("修正内容を準備しています");
  $("edit-history").insertAdjacentHTML("beforeend", `<div class="user">あなた: ${escapeHtml(instruction)}</div>`);
  try {
    const maskFile = $("edit-mask").files[0];
    const maskImage = maskFile ? await fileAsDataUrl(maskFile) : "";
    const data = await runJob({
      mode: "edit",
      prompt: instruction,
      source_image: currentImage,
      mask_image: maskImage,
      ...generationSettings(),
    });
    editInstructions.push(instruction);
    showResult(data);
    $("edit-history").insertAdjacentHTML("beforeend", "<div>AI: 修正画像を生成しました。</div>");
    $("edit-prompt").value = "";
    $("edit-mask").value = "";
  } catch (error) {
    setError(error.message);
    finishProgress();
  } finally {
    button.disabled = false;
  }
});

$("save-r2").addEventListener("click", async () => {
  if (!currentImage) return setError("保存できる画像がありません。");
  const button = $("save-r2");
  button.disabled = true;
  try {
    const response = await fetch("/api/save-to-r2", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({image: currentImage, metadata: currentMetadata}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "R2保存に失敗しました");
    button.textContent = "R2に保存済み";
  } catch (error) {
    setError(error.message);
  } finally {
    button.disabled = false;
  }
});

$("download").addEventListener("click", () => {
  if (!currentImage) return;
  const link = document.createElement("a");
  link.href = "data:image/png;base64," + currentImage;
  link.download = `janku-${Date.now()}.png`;
  link.click();
});

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
}
