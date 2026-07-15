const $ = (id) => document.getElementById(id);
const presets = JSON.parse($("preset-data").textContent);
const styleFields = [
  ["anime-strength", "anime_strength"],
  ["line-detail", "line_detail"],
  ["color-vividness", "color_vividness"],
  ["photoreal-avoidance", "photoreal_avoidance"],
];

const selectedCatalog = {character: [], compose: []};
const currentImages = {character: "", compose: ""};
const catalogTimers = {character: null, compose: null};
const catalogRequestVersion = {character: 0, compose: 0};
const catalogViews = {
  character: {items: [], next_offset: 0, has_more: false, total: 0},
  compose: {items: [], next_offset: 0, has_more: false, total: 0},
};

let activeWorkflow = "character";
let composeSourceImage = "";
let composeSourceLabel = "";
let compositionForegroundImage = "";
let compositionBackgroundMask = "";
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

function catalogId(workflow, suffix) {
  if (suffix === "selected") return `${workflow}-selected`;
  return `${workflow}-catalog-${suffix}`;
}

function promptId(workflow) {
  return `${workflow}-prompt`;
}

function refineId(workflow) {
  return `${workflow}-refine-enabled`;
}

function setWorkflow(workflow) {
  activeWorkflow = workflow;
  const character = workflow === "character";
  $("panel-character").hidden = !character;
  $("panel-compose").hidden = character;
  $("tab-character").setAttribute("aria-pressed", String(character));
  $("tab-compose").setAttribute("aria-pressed", String(!character));
  if (!character && currentImages.character && !composeSourceImage) {
    setComposeSource(currentImages.character, "キャラクター生成の結果");
  }
}

function promptForWorkflow(workflow) {
  const freeText = $(promptId(workflow)).value.trim();
  const catalogPrompt = selectedCatalog[workflow].map((item) => item.prompt).join(", ");
  return [catalogPrompt, freeText].filter(Boolean).join("\n");
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function renderSelected(workflow) {
  const container = $(catalogId(workflow, "selected"));
  container.replaceChildren();
  selectedCatalog[workflow].forEach((item) => {
    const chip = createElement("button", "prompt-chip", `${item.name}: ${item.prompt}`);
    chip.type = "button";
    chip.title = "選択から削除";
    chip.addEventListener("click", () => {
      selectedCatalog[workflow] = selectedCatalog[workflow].filter((selected) => selected.id !== item.id);
      renderSelected(workflow);
    });
    container.append(chip);
  });
}

function renderCatalog(workflow, data) {
  const category = $(catalogId(workflow, "category"));
  if (!category.options.length) {
    const all = new Option("すべてのカテゴリ", "");
    category.add(all);
    data.categories.forEach((item) => category.add(new Option(`${item.title} (${item.count})`, item.id)));
  }

  const subcategory = $(catalogId(workflow, "subcategory"));
  const selectedSubcategory = subcategory.value;
  subcategory.replaceChildren(new Option("すべての中分類", ""));
  (data.subcategories || []).forEach((item) => subcategory.add(new Option(item.title, item.id)));
  subcategory.disabled = !category.value || !data.subcategories?.length;
  if (selectedSubcategory && [...subcategory.options].some((option) => option.value === selectedSubcategory)) {
    subcategory.value = selectedSubcategory;
  }

  const container = $(catalogId(workflow, "results"));
  container.replaceChildren();
  const more = $(catalogId(workflow, "more"));
  more.hidden = !data.has_more;
  if (data.has_more) {
    more.textContent = `さらに読み込む（残り ${data.total - data.next_offset} 件）`;
  }
  if (!data.items.length) {
    container.append(createElement("div", "catalog-empty", "該当する項目がありません"));
    return;
  }
  data.items.forEach((item) => {
    const row = createElement("button", "catalog-item");
    row.type = "button";
    const isSelected = selectedCatalog[workflow].some((selected) => selected.id === item.id);
    row.classList.toggle("is-selected", isSelected);
    row.setAttribute("aria-pressed", String(isSelected));
    row.title = isSelected ? "選択済み。もう一度押すと選択を解除します" : "この項目をプロンプトへ追加";
    const name = createElement("strong", "catalog-item-name", item.name);
    const prompt = createElement("span", "catalog-item-prompt", item.prompt);
    row.append(name, prompt);
    if (item.description) row.append(createElement("span", "catalog-item-description", item.description));
    row.addEventListener("click", () => {
      const existing = selectedCatalog[workflow].findIndex((selected) => selected.id === item.id);
      if (existing >= 0) {
        selectedCatalog[workflow].splice(existing, 1);
        renderSelected(workflow);
        renderCatalog(workflow, catalogViews[workflow]);
        return;
      }
      if (selectedCatalog[workflow].length >= 24) {
        setError("カタログから選べる項目は24件までです。");
        return;
      }
      selectedCatalog[workflow].push(item);
      renderSelected(workflow);
      renderCatalog(workflow, catalogViews[workflow]);
      setError();
    });
    container.append(row);
  });
}

async function loadCatalog(workflow, append = false) {
  const requestVersion = ++catalogRequestVersion[workflow];
  const category = $(catalogId(workflow, "category")).value;
  const subcategory = $(catalogId(workflow, "subcategory")).value;
  const query = $(catalogId(workflow, "query")).value.trim();
  const params = new URLSearchParams({limit: "72"});
  if (category) params.set("category", category);
  if (subcategory) params.set("subcategory", subcategory);
  if (query) params.set("q", query);
  if (append) params.set("offset", String(catalogViews[workflow].next_offset || 0));
  try {
    const response = await fetch(`/api/prompt-catalog?${params}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "プロンプトカタログを読み込めませんでした");
    if (requestVersion !== catalogRequestVersion[workflow]) return;
    catalogViews[workflow] = {
      ...data,
      items: append ? [...catalogViews[workflow].items, ...data.items] : data.items,
    };
    renderCatalog(workflow, catalogViews[workflow]);
  } catch (error) {
    if (requestVersion === catalogRequestVersion[workflow]) setError(error.message);
  }
}

function queueCatalogSearch(workflow) {
  clearTimeout(catalogTimers[workflow]);
  catalogTimers[workflow] = setTimeout(() => loadCatalog(workflow), 180);
}

function fileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setComposeSource(image, label, {preserveForeground = false} = {}) {
  composeSourceImage = image || "";
  composeSourceLabel = label || "";
  if (!preserveForeground) {
    compositionForegroundImage = composeSourceImage;
    compositionBackgroundMask = "";
  }
  const hasSource = Boolean(composeSourceImage);
  $("compose-source-preview").hidden = !hasSource;
  $("compose-source-empty").hidden = hasSource;
  if (hasSource) {
    $("compose-source-image").src = `data:image/png;base64,${composeSourceImage}`;
    $("compose-source-image").alt = composeSourceLabel || "編集元の画像";
  }
}

function addEditHistory(instruction) {
  const line = createElement("div", "user", `あなた: ${instruction}`);
  $("edit-history").append(line);
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
  $("empty").hidden = true;
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
        $("progress-fill").style.width = `${percent}%`;
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
  currentImages[data.workflow] = data.image;
  currentMetadata = {
    original_prompt: rootPrompt || data.original_prompt,
    edit_history: [...editInstructions],
    optimized_prompt: data.optimized_prompt,
    optimizer_source: data.optimizer_source,
    intent_notes: data.intent_notes,
    refine_enabled: data.refine_enabled,
    generation_settings: data.settings,
    workflow: data.workflow,
    editor_model: data.editor_model || "",
    edit_strength: data.edit_strength ?? null,
    edit_scope: data.edit_scope || "",
    background_mask: data.background_mask || "",
  };
  $("result-image").src = `data:image/png;base64,${data.image}`;
  $("optimized-prompt").textContent = data.optimized_prompt;
  $("result-workflow-label").textContent = data.workflow === "character" ? "キャラクター生成" : "背景合成 / 修正";
  $("use-result-in-compose").hidden = data.workflow !== "character";
  $("save-r2").textContent = "R2に保存";
  if (data.workflow === "character") setComposeSource(data.image, "キャラクター生成の結果");
  if (data.workflow === "compose") {
    if (data.background_mask) compositionBackgroundMask = data.background_mask;
    setComposeSource(data.image, "直前の修正結果", {
      preserveForeground: Boolean(data.background_mask),
    });
  }
  finishProgress();
}

async function generateCharacter() {
  const prompt = promptForWorkflow("character");
  if (!prompt) return setError("キャラクターの要素を入力するか、カタログから選択してください。");
  const button = $("generate-character");
  button.disabled = true;
  beginProgress("キャラクター設定を確認しています");
  try {
    const data = await runJob({
      workflow: "character",
      mode: "t2i",
      prompt,
      refine_enabled: $("character-refine-enabled").checked,
      ...generationSettings(),
    });
    rootPrompt = prompt;
    editInstructions = [];
    $("edit-history").replaceChildren();
    showResult(data);
  } catch (error) {
    setError(error.message);
    $("progress").hidden = true;
    $("empty").hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function generateCompose() {
  const prompt = promptForWorkflow("compose");
  if (!composeSourceImage) return setError("編集元の画像を選択してください。");
  if (!prompt) return setError("背景・構図・修正内容を入力するか、カタログから選択してください。");
  const button = $("generate-compose");
  button.disabled = true;
  beginProgress("背景と構図の指示を準備しています");
  addEditHistory(prompt);
  try {
    const maskFile = $("compose-mask").files[0];
    const maskImage = maskFile ? await fileAsDataUrl(maskFile) : "";
    const editScope = $("compose-edit-scope").value;
    const useSceneContext = Boolean(
      editScope === "background"
      && !maskImage
      && compositionForegroundImage
      && compositionBackgroundMask
    );
    const requestPrompt = editScope === "background" && editInstructions.length
      ? [...editInstructions, prompt].join("\n")
      : prompt;
    const data = await runJob({
      workflow: "compose",
      mode: "edit",
      prompt: requestPrompt,
      source_image: useSceneContext ? compositionForegroundImage : composeSourceImage,
      mask_image: maskImage,
      editor_model: "waifu_inpaint_xl",
      edit_scope: editScope,
      background_mask_image: useSceneContext ? compositionBackgroundMask : "",
      edit_strength: Number($("compose-strength").value) / 100,
      refine_enabled: $("compose-refine-enabled").checked,
      ...generationSettings(),
    });
    rootPrompt = rootPrompt || prompt;
    editInstructions.push(prompt);
    showResult(data);
    $("edit-history").append(createElement("div", "assistant", "AI: 修正画像を生成しました。"));
    $("compose-prompt").value = "";
    $("compose-mask").value = "";
  } catch (error) {
    setError(error.message);
    $("progress").hidden = true;
    if (currentImage) $("result").hidden = false;
  } finally {
    button.disabled = false;
  }
}

async function saveToR2() {
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
}

$("preset").addEventListener("change", applyPreset);
styleFields.forEach(([id]) => $(id).addEventListener("input", () => syncSlider(id)));
$("compose-strength").addEventListener("input", () => {
  $("compose-strength-value").value = $("compose-strength").value;
});
function syncComposeEditControls() {
  const isBackground = $("compose-edit-scope").value === "background";
  $("compose-strength-field").hidden = isBackground;
  $("compose-strength").value = isBackground ? 85 : 55;
  $("compose-strength-value").value = $("compose-strength").value;
}

$("compose-edit-scope").addEventListener("change", syncComposeEditControls);
syncComposeEditControls();
$("generate-character").addEventListener("click", generateCharacter);
$("generate-compose").addEventListener("click", generateCompose);
$("save-r2").addEventListener("click", saveToR2);
$("download").addEventListener("click", () => {
  if (!currentImage) return;
  const link = document.createElement("a");
  link.href = `data:image/png;base64,${currentImage}`;
  link.download = `illustration-${Date.now()}.png`;
  link.click();
});
$("use-character-result").addEventListener("click", () => {
  if (currentImages.character) setComposeSource(currentImages.character, "キャラクター生成の結果");
});
$("use-result-in-compose").addEventListener("click", () => setWorkflow("compose"));
$("compose-source-file").addEventListener("change", async () => {
  const file = $("compose-source-file").files[0];
  if (!file) return;
  try {
    const dataUrl = await fileAsDataUrl(file);
    setComposeSource(dataUrl.slice(dataUrl.indexOf(",") + 1), file.name);
  } catch {
    setError("画像ファイルを読み込めませんでした。");
  }
});

["character", "compose"].forEach((workflow) => {
  $(catalogId(workflow, "category")).addEventListener("change", () => {
    $(catalogId(workflow, "subcategory")).value = "";
    loadCatalog(workflow);
  });
  $(catalogId(workflow, "subcategory")).addEventListener("change", () => loadCatalog(workflow));
  $(catalogId(workflow, "query")).addEventListener("input", () => queueCatalogSearch(workflow));
  $(catalogId(workflow, "more")).addEventListener("click", () => loadCatalog(workflow, true));
  loadCatalog(workflow);
});
document.querySelectorAll(".workflow-tab").forEach((button) => {
  button.addEventListener("click", () => setWorkflow(button.dataset.workflow));
});

applyPreset();
setWorkflow("character");
