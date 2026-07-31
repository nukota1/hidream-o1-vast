const $ = (id) => document.getElementById(id);
const presets = JSON.parse($("preset-data").textContent);
const styleFields = [
  ["anime-strength", "anime_strength"],
  ["line-detail", "line_detail"],
  ["color-vividness", "color_vividness"],
  ["photoreal-avoidance", "photoreal_avoidance"],
];
const catalogWorkflows = ["character", "compose", "settings"];
const selectedCatalog = {character: [], compose: []};
const selectedTabs = {character: "", compose: ""};
const currentImages = {character: "", compose: ""};
const catalogTimers = {character: null, compose: null, settings: null};
const catalogRequestVersion = {character: 0, compose: 0, settings: 0};
const catalogViews = Object.fromEntries(catalogWorkflows.map((workflow) => [workflow, {
  categories: [], subcategories: [], items: [], next_offset: 0, has_more: false, total: 0,
}]));

const DB_NAME = "illustration-studio";
const DB_VERSION = 2;
const HISTORY_STORE = "prompt-history";
const FAVORITES_STORE = "favorite-groups";
const FOLDERS_STORE = "gallery-folders";
const STYLE_CACHE_KEY = "illustration-style-strength-v1";

let activeWorkflow = "character";
let composeSourceImage = "";
let composeSourceLabel = "";
let currentImage = "";
let currentMetadata = {};
let batchResults = [];
let selectedBatchIndex = 0;
let rootPrompt = "";
let rootOptimizedPrompt = "";
let rootCharacterPrompt = "";
let rootOptimizedCharacterPrompt = "";
let rootScenePrompt = "";
let rootOptimizedScenePrompt = "";
let rootLoraMetadata = {};
let editInstructions = [];
let historyTargetWorkflow = "character";
let favoriteGroups = [];
let activeFavoriteGroupId = "";
let galleryFolders = [];
let galleryRecords = [];
let activeGalleryFolderId = "";
let movingGalleryRecordId = "";
let galleryBackend = "local";
let currentGalleryRecordId = "";
let galleryContextRecordId = "";
let loraModels = [];
let currentLoraModelType = "";
let loraImageLimits = {character: 100, style: 150, pose: 100, background: 150};
let loraRecommendedCounts = {character: 10, style: 50, pose: 20, background: 30};
let loraTrainingSelectedIds = new Set();
let loraTrainingUploads = [];

function setError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function createElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = text;
  return element;
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(HISTORY_STORE)) {
        const history = db.createObjectStore(HISTORY_STORE, {keyPath: "id"});
        history.createIndex("createdAt", "createdAt");
      }
      if (!db.objectStoreNames.contains(FAVORITES_STORE)) {
        const favorites = db.createObjectStore(FAVORITES_STORE, {keyPath: "id"});
        favorites.createIndex("createdAt", "createdAt");
      }
      if (!db.objectStoreNames.contains(FOLDERS_STORE)) {
        const folders = db.createObjectStore(FOLDERS_STORE, {keyPath: "id"});
        folders.createIndex("createdAt", "createdAt");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function storeGetAll(storeName) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const request = db.transaction(storeName, "readonly").objectStore(storeName).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).finally(() => db.close());
}

async function storePut(storeName, value) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, "readwrite");
    transaction.objectStore(storeName).put(value);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
  }).finally(() => db.close());
}

async function storeDelete(storeName, key) {
  const db = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, "readwrite");
    transaction.objectStore(storeName).delete(key);
    transaction.oncomplete = resolve;
    transaction.onerror = () => reject(transaction.error);
  }).finally(() => db.close());
}

async function detectGalleryBackend() {
  try {
    const response = await fetch("/api/gallery/health", {cache: "no-store"});
    if (response.status === 404) {
      galleryBackend = "local";
      return;
    }
    const data = await response.json();
    if (!response.ok || data.storage !== "cloudflare") {
      throw new Error(data.error || "Cloudflare storage is not ready.");
    }
    galleryBackend = "cloud";
  } catch (error) {
    if (galleryBackend === "cloud") throw error;
    galleryBackend = "local";
  }
}

async function cloudRequest(path, options = {}) {
  if (galleryBackend !== "cloud") return null;
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "Cloudflare storage request failed.");
  return data;
}

function galleryImageSource(record) {
  if (record.image_url) return record.image_url;
  if (record.image) return `data:image/png;base64,${record.image}`;
  return "";
}

async function backendRequest(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || "サーバー処理に失敗しました。");
  return data;
}

function syncLoraPickers(preferredId = "") {
  const preferredModel = loraModels.find((model) => model.id === preferredId);
  ["character", "compose"].forEach((workflow) => {
    [["character", `${workflow}-lora-select`], ["style", `${workflow}-style-lora-select`]]
      .forEach(([category, selectId]) => {
        const select = $(selectId);
        const previous = preferredModel?.category === category ? preferredId : select.value;
        select.replaceChildren(new Option("使用しない", ""));
        loraModels
          .filter((model) => (
            model.status === "ready"
            && model.category === category
            && (model.compatible ?? model.model_type === currentLoraModelType)
          ))
          .forEach((model) => {
            select.add(new Option(`${model.name}（${model.trigger_word}）`, model.id));
          });
        select.value = [...select.options].some((option) => option.value === previous) ? previous : "";
      });
  });
  const training = loraModels.filter((model) => model.status === "training" || model.status === "queued").length;
  const failed = loraModels.filter((model) => model.status === "failed").length;
  const incompatible = loraModels.filter((model) => model.status === "ready" && model.compatible === false).length;
  $("character-lora-help").textContent = training
    ? `${training}件のLoRAを学習中です。生成処理とは順番に実行されます。`
    : failed
      ? `使用可能 ${loraModels.filter((model) => model.status === "ready").length}件 / 失敗 ${failed}件`
      : incompatible
        ? `${incompatible}件の旧基盤モデル用LoRAは互換性保護のため非表示です。`
        : "ギャラリー画像のメニューから追加学習できます。";
}

async function loadLoraModels(preferredId = "") {
  const data = await backendRequest("/api/lora/models");
  loraModels = data.items || [];
  currentLoraModelType = data.current_model_type || "";
  loraImageLimits = {...loraImageLimits, ...(data.max_image_counts || {})};
  loraRecommendedCounts = {...loraRecommendedCounts, ...(data.recommended_image_counts || {})};
  const modelLabel = data.current_model_label || currentLoraModelType;
  $("lora-model-type").replaceChildren(new Option(modelLabel, currentLoraModelType));
  syncLoraPickers(preferredId);
  syncLoraTrainingCategory();
}

function loraGenerationSettings(workflow) {
  const characterId = $(`${workflow}-lora-select`).value;
  const styleId = $(`${workflow}-style-lora-select`).value;
  return {
    ...(characterId ? {
      character_lora_id: characterId,
      character_lora_weight: Number($(`${workflow}-lora-weight`).value) / 100,
    } : {}),
    ...(styleId ? {
      style_lora_id: styleId,
      style_lora_weight: Number($(`${workflow}-style-lora-weight`).value) / 100,
    } : {}),
  };
}

function loraMetadataFromResult(data) {
  const characterId = data.character_lora_id || data.lora_id || "";
  return {
    character_lora_id: characterId,
    character_lora_name: data.character_lora_name || data.lora_name || "",
    character_lora_trigger_word: data.character_lora_trigger_word || data.lora_trigger_word || "",
    character_lora_weight: data.character_lora_weight ?? data.lora_weight ?? null,
    style_lora_id: data.style_lora_id || "",
    style_lora_name: data.style_lora_name || "",
    style_lora_trigger_word: data.style_lora_trigger_word || "",
    style_lora_weight: data.style_lora_weight ?? null,
  };
}

function applyLoraMetadataToWorkflow(metadata, workflow) {
  const characterId = metadata.character_lora_id || metadata.lora_id || "";
  if (characterId && [...$(`${workflow}-lora-select`).options].some(
    (option) => option.value === characterId
  )) {
    $(`${workflow}-lora-select`).value = characterId;
    const weight = metadata.character_lora_weight ?? metadata.lora_weight;
    if (weight != null) {
      $(`${workflow}-lora-weight`).value = Math.round(Number(weight) * 100);
      $(`${workflow}-lora-weight-value`).value = $(`${workflow}-lora-weight`).value;
    }
  }
  const styleId = metadata.style_lora_id || "";
  if (styleId && [...$(`${workflow}-style-lora-select`).options].some(
    (option) => option.value === styleId
  )) {
    $(`${workflow}-style-lora-select`).value = styleId;
    if (metadata.style_lora_weight != null) {
      $(`${workflow}-style-lora-weight`).value = Math.round(Number(metadata.style_lora_weight) * 100);
      $(`${workflow}-style-lora-weight-value`).value = $(`${workflow}-style-lora-weight`).value;
    }
  }
}

async function persistFavoriteGroup(group) {
  if (galleryBackend === "cloud") {
    await cloudRequest(`/api/favorites/${encodeURIComponent(group.id)}`, {
      method: "PUT",
      body: JSON.stringify(group),
    });
    return;
  }
  await storePut(FAVORITES_STORE, group);
}

async function removeFavoriteGroup(id) {
  if (galleryBackend === "cloud") {
    await cloudRequest(`/api/favorites/${encodeURIComponent(id)}`, {method: "DELETE"});
    return;
  }
  await storeDelete(FAVORITES_STORE, id);
}

function syncSlider(id) {
  $(id + "-value").value = $(id).value;
}

function persistStyleSettings() {
  try {
    localStorage.setItem(STYLE_CACHE_KEY, JSON.stringify(styleSettings()));
  } catch (error) {
    console.warn("Style settings could not be cached", error);
  }
}

function applyStyleSettings(values, persist = true) {
  if (!values || typeof values !== "object") return false;
  let applied = false;
  styleFields.forEach(([id, key]) => {
    const value = Number(values[key]);
    if (!Number.isFinite(value)) return;
    $(id).value = Math.max(0, Math.min(100, Math.round(value)));
    syncSlider(id);
    applied = true;
  });
  if (applied && persist) persistStyleSettings();
  return applied;
}

function restoreCachedStyleSettings() {
  try {
    const cached = JSON.parse(localStorage.getItem(STYLE_CACHE_KEY) || "null");
    applyStyleSettings(cached, false);
  } catch (error) {
    console.warn("Cached style settings could not be restored", error);
  }
}

function applyPreset(persist = true) {
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
  if (persist) persistStyleSettings();
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
  if (suffix === "selected-tabs") return `${workflow}-selected-tabs`;
  return `${workflow}-catalog-${suffix}`;
}

function promptId(workflow) {
  return `${workflow}-prompt`;
}

function setWorkflow(workflow) {
  activeWorkflow = workflow;
  ["character", "compose", "gallery", "settings"].forEach((name) => {
    $(`panel-${name}`).hidden = name !== workflow;
    $(`tab-${name}`).setAttribute("aria-pressed", String(name === workflow));
  });
  $("result-area").hidden = workflow === "gallery";
  $("gallery-area").hidden = workflow !== "gallery";
  if (workflow === "compose" && currentImages.character && !composeSourceImage) {
    setComposeSource(currentImages.character, "キャラクター生成の結果");
    $("compose-character-definition").value = rootCharacterPrompt || rootOptimizedCharacterPrompt;
  }
  if (workflow === "settings") renderFavoriteSettings();
  if (workflow === "gallery") loadGallery();
}

function promptPartsForWorkflow(workflow) {
  const freeText = $(promptId(workflow)).value.trim();
  const catalogPrompt = selectedCatalog[workflow].map((item) => item.prompt).join(", ");
  // Free text expresses the user's exact intent, so it must reach Qwen and
  // CLIP before the optional catalog tags when the prompt needs compacting.
  return {
    freeText,
    catalogPrompt,
    prompt: [freeText, catalogPrompt].filter(Boolean).join("\n"),
  };
}

function batchCountForWorkflow(workflow) {
  const input = $(`${workflow}-batch-count`);
  if (!input) return 1;
  const count = Math.max(1, Math.min(8, Math.round(Number(input.value) || 1)));
  input.value = count;
  return count;
}

function selectedGroupName(item) {
  return item.subcategory_title || item.group || "その他";
}

function renderSelected(workflow) {
  const container = $(catalogId(workflow, "selected"));
  const tabs = $(catalogId(workflow, "selected-tabs"));
  const box = container.closest(".selected-box");
  const groups = [...new Set(selectedCatalog[workflow].map(selectedGroupName))];
  box.classList.toggle("is-empty", groups.length === 0);
  tabs.replaceChildren();
  container.replaceChildren();
  if (!groups.length) {
    selectedTabs[workflow] = "";
    return;
  }
  if (!groups.includes(selectedTabs[workflow])) selectedTabs[workflow] = groups[0];
  groups.forEach((group) => {
    const count = selectedCatalog[workflow].filter((item) => selectedGroupName(item) === group).length;
    const tab = createElement("button", "selected-tab", `${group} (${count})`);
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(group === selectedTabs[workflow]));
    tab.addEventListener("click", () => {
      selectedTabs[workflow] = group;
      renderSelected(workflow);
    });
    tabs.append(tab);
  });
  selectedCatalog[workflow]
    .filter((item) => selectedGroupName(item) === selectedTabs[workflow])
    .forEach((item) => {
      const chip = createElement("button", "prompt-chip", `${item.name}: ${item.prompt}`);
      chip.type = "button";
      chip.title = "選択から削除";
      chip.addEventListener("click", () => {
        selectedCatalog[workflow] = selectedCatalog[workflow].filter((selected) => selected.id !== item.id);
        renderSelected(workflow);
        renderCatalog(workflow, catalogViews[workflow]);
      });
      container.append(chip);
    });
}

function refreshCategoryOptions(workflow, categories) {
  const select = $(catalogId(workflow, "category"));
  const previous = select.value;
  select.replaceChildren(new Option("すべてのカテゴリ", ""));
  categories.forEach((item) => select.add(new Option(`${item.title} (${item.count})`, item.id)));
  if (["character", "compose"].includes(workflow) && favoriteGroups.length) {
    select.add(new Option("お気に入り", "__favorites__"));
  }
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}

function refreshSubcategoryOptions(workflow, subcategories) {
  const select = $(catalogId(workflow, "subcategory"));
  const previous = select.value;
  select.replaceChildren(new Option("すべての中分類", ""));
  subcategories.forEach((item) => select.add(new Option(item.title, item.id)));
  select.disabled = !subcategories.length;
  if ([...select.options].some((option) => option.value === previous)) select.value = previous;
}

function activeFavoriteGroup() {
  return favoriteGroups.find((group) => group.id === activeFavoriteGroupId) || null;
}

function refreshFavoriteCategories() {
  ["character", "compose"].forEach((workflow) => {
    const wasViewingFavorites = $(catalogId(workflow, "category")).value === "__favorites__";
    refreshCategoryOptions(workflow, catalogViews[workflow].categories);
    if (wasViewingFavorites) loadCatalog(workflow);
  });
}

function isCatalogItemSelected(workflow, item) {
  if (workflow === "settings") return Boolean(activeFavoriteGroup()?.items.some((entry) => entry.id === item.id));
  return selectedCatalog[workflow].some((entry) => entry.id === item.id);
}

async function toggleFavoriteItem(item) {
  const group = activeFavoriteGroup();
  if (!group) {
    setError("先にお気に入りグループを作成してください。");
    return;
  }
  const index = group.items.findIndex((entry) => entry.id === item.id);
  if (index >= 0) group.items.splice(index, 1);
  else group.items.push(item);
  group.updatedAt = Date.now();
  await persistFavoriteGroup(group);
  renderFavoriteSettings();
  renderCatalog("settings", catalogViews.settings);
  refreshFavoriteCategories();
  setError();
}

function renderCatalog(workflow, data) {
  refreshCategoryOptions(workflow, data.categories || []);
  refreshSubcategoryOptions(workflow, data.subcategories || []);
  const container = $(catalogId(workflow, "results"));
  const more = $(catalogId(workflow, "more"));
  if (workflow !== "settings") {
    const favoritesSelected = $(catalogId(workflow, "category")).value === "__favorites__";
    $(`${workflow}-favorite-import`).hidden = !favoritesSelected;
    $(`${workflow}-catalog-reset`).disabled = selectedCatalog[workflow].length === 0;
  }
  container.replaceChildren();
  more.hidden = !data.has_more;
  if (data.has_more) more.textContent = `さらに読み込む（残り ${data.total - data.next_offset} 件）`;
  if (!data.items.length) {
    container.append(createElement("div", "catalog-empty", "該当する項目がありません"));
    return;
  }
  data.items.forEach((item) => {
    const row = createElement("button", "catalog-item");
    row.type = "button";
    const selected = isCatalogItemSelected(workflow, item);
    row.classList.toggle("is-selected", selected);
    row.setAttribute("aria-pressed", String(selected));
    row.title = workflow === "settings"
      ? (selected ? "お気に入りから解除" : "編集中のグループへ登録")
      : (selected ? "選択済み。もう一度押すと解除" : "プロンプトへ追加");
    row.append(
      createElement("strong", "catalog-item-name", item.name),
      createElement("span", "catalog-item-prompt", item.prompt),
    );
    if (item.description) row.append(createElement("span", "catalog-item-description", item.description));
    row.addEventListener("click", async () => {
      if (workflow === "settings") {
        await toggleFavoriteItem(item);
        return;
      }
      const index = selectedCatalog[workflow].findIndex((entry) => entry.id === item.id);
      if (index >= 0) selectedCatalog[workflow].splice(index, 1);
      else selectedCatalog[workflow].push(item);
      selectedTabs[workflow] = selectedGroupName(item);
      renderSelected(workflow);
      renderCatalog(workflow, catalogViews[workflow]);
      setError();
    });
    container.append(row);
  });
}

function favoriteCatalogData(workflow, query, groupId) {
  const terms = query.toLocaleLowerCase();
  const groups = groupId ? favoriteGroups.filter((group) => group.id === groupId) : favoriteGroups;
  const deduplicated = new Map();
  groups.forEach((group) => group.items.forEach((item) => {
    const haystack = `${item.name} ${item.prompt} ${item.description || ""}`.toLocaleLowerCase();
    if (!terms || haystack.includes(terms)) deduplicated.set(item.id, item);
  }));
  return {
    categories: catalogViews[workflow].categories,
    subcategories: favoriteGroups.map((group) => ({id: group.id, title: group.name})),
    items: [...deduplicated.values()],
    total: deduplicated.size,
    next_offset: deduplicated.size,
    has_more: false,
  };
}

function importFavoriteGroup(workflow) {
  const groupId = $(catalogId(workflow, "subcategory")).value;
  const sourceGroups = groupId
    ? favoriteGroups.filter((group) => group.id === groupId)
    : favoriteGroups;
  const existingIds = new Set(selectedCatalog[workflow].map((item) => item.id));
  const additions = [];
  sourceGroups.forEach((group) => group.items.forEach((item) => {
    if (!existingIds.has(item.id)) {
      existingIds.add(item.id);
      additions.push(item);
    }
  }));
  selectedCatalog[workflow].push(...additions);
  if (additions.length) selectedTabs[workflow] = selectedGroupName(additions[0]);
  renderSelected(workflow);
  renderCatalog(workflow, catalogViews[workflow]);
  setError();
}

function resetCatalogSelection(workflow) {
  selectedCatalog[workflow] = [];
  selectedTabs[workflow] = "";
  renderSelected(workflow);
  renderCatalog(workflow, catalogViews[workflow]);
  setError();
}

async function loadCatalog(workflow, append = false) {
  const category = $(catalogId(workflow, "category")).value;
  const subcategory = $(catalogId(workflow, "subcategory")).value;
  const query = $(catalogId(workflow, "query")).value.trim();
  if (["character", "compose"].includes(workflow) && category === "__favorites__") {
    catalogViews[workflow] = favoriteCatalogData(workflow, query, subcategory);
    renderCatalog(workflow, catalogViews[workflow]);
    return;
  }
  const requestVersion = ++catalogRequestVersion[workflow];
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

function renderFavoriteSettings() {
  const select = $("favorite-group-select");
  if (!favoriteGroups.some((group) => group.id === activeFavoriteGroupId)) {
    activeFavoriteGroupId = favoriteGroups[0]?.id || "";
  }
  select.replaceChildren();
  if (!favoriteGroups.length) select.add(new Option("グループがありません", ""));
  favoriteGroups.forEach((group) => select.add(new Option(`${group.name} (${group.items.length})`, group.id)));
  select.value = activeFavoriteGroupId;
  select.disabled = !favoriteGroups.length;
  $("favorite-group-rename").disabled = !favoriteGroups.length;
  $("favorite-group-delete").disabled = !favoriteGroups.length;

  const summary = $("favorite-summary");
  summary.replaceChildren();
  const group = activeFavoriteGroup();
  if (!group?.items.length) {
    summary.append(createElement("div", "favorite-summary-empty", "このグループにはまだ項目がありません。"));
    return;
  }
  group.items.forEach((item) => {
    const row = createElement("div", "favorite-summary-item");
    const label = createElement("span", "", `${item.name}: ${item.prompt}`);
    const remove = createElement("button", "", "解除");
    remove.type = "button";
    remove.addEventListener("click", () => toggleFavoriteItem(item));
    row.append(label, remove);
    summary.append(row);
  });
}

async function createFavoriteGroup() {
  const input = $("favorite-group-name");
  const name = input.value.trim() || `お気に入り${favoriteGroups.length + 1}`;
  const group = {id: crypto.randomUUID(), name, items: [], createdAt: Date.now(), updatedAt: Date.now()};
  await persistFavoriteGroup(group);
  favoriteGroups.push(group);
  activeFavoriteGroupId = group.id;
  input.value = "";
  renderFavoriteSettings();
  renderCatalog("settings", catalogViews.settings);
  refreshFavoriteCategories();
}

async function addCustomFavoritePrompt() {
  const group = activeFavoriteGroup();
  if (!group) return setError("先に登録先のお気に入りグループを作成してください。");
  const name = $("custom-prompt-name").value.trim();
  const prompt = $("custom-prompt-value").value.trim();
  const description = $("custom-prompt-description").value.trim();
  if (!name || !prompt) return setError("日本語名とプロンプトを入力してください。");
  group.items.push({
    id: `custom:${crypto.randomUUID()}`,
    category: "custom",
    subcategory: "favorite-custom",
    subcategory_title: "ユーザー登録",
    group: "ユーザー登録",
    name,
    prompt,
    description,
  });
  group.updatedAt = Date.now();
  await persistFavoriteGroup(group);
  $("custom-prompt-name").value = "";
  $("custom-prompt-value").value = "";
  $("custom-prompt-description").value = "";
  renderFavoriteSettings();
  refreshFavoriteCategories();
  setError();
}

async function renameFavoriteGroup() {
  const group = activeFavoriteGroup();
  if (!group) return;
  const name = window.prompt("お気に入りグループの新しい名前", group.name)?.trim();
  if (!name || name === group.name) return;
  group.name = name;
  group.updatedAt = Date.now();
  await persistFavoriteGroup(group);
  renderFavoriteSettings();
  refreshFavoriteCategories();
}

async function deleteFavoriteGroup() {
  const group = activeFavoriteGroup();
  if (!group || !window.confirm(`「${group.name}」を削除しますか？`)) return;
  await removeFavoriteGroup(group.id);
  favoriteGroups = favoriteGroups.filter((entry) => entry.id !== group.id);
  activeFavoriteGroupId = favoriteGroups[0]?.id || "";
  renderFavoriteSettings();
  renderCatalog("settings", catalogViews.settings);
  refreshFavoriteCategories();
}

async function savePromptHistory(data, prompt) {
  if (!data.image || !prompt) return;
  const record = {
    id: crypto.randomUUID(),
    createdAt: Date.now(),
    workflow: data.workflow,
    image: data.image,
    prompt,
    optimizedPrompt: data.optimized_prompt || "",
    folderId: "",
    metadata: {
      original_prompt: data.original_prompt || prompt,
      free_prompt: data.free_prompt || "",
      catalog_prompt: data.catalog_prompt || "",
      optimized_prompt: data.optimized_prompt || "",
      optimizer_source: data.optimizer_source || "",
      intent_notes: data.intent_notes || "",
      generation_settings: data.settings || {},
      style_settings: data.settings?.style || styleSettings(),
      image_model_profile: data.image_model_profile || "",
      image_model_label: data.image_model_label || "",
      workflow: data.workflow || "",
      editor_model: data.editor_model || "",
      edit_strength: data.edit_strength ?? null,
      edit_scope: data.edit_scope || "",
      lora_id: data.lora_id || "",
      lora_name: data.lora_name || "",
      lora_trigger_word: data.lora_trigger_word || "",
      lora_weight: data.lora_weight ?? null,
      character_lora_id: data.character_lora_id || data.lora_id || "",
      character_lora_name: data.character_lora_name || data.lora_name || "",
      character_lora_trigger_word: data.character_lora_trigger_word || data.lora_trigger_word || "",
      character_lora_weight: data.character_lora_weight ?? data.lora_weight ?? null,
      style_lora_id: data.style_lora_id || "",
      style_lora_name: data.style_lora_name || "",
      style_lora_trigger_word: data.style_lora_trigger_word || "",
      style_lora_weight: data.style_lora_weight ?? null,
      applied_loras: data.applied_loras || [],
      generation_intent: data.generation_intent || "",
      character_prompt: data.character_prompt || "",
      scene_prompt: data.scene_prompt || "",
      optimized_character_prompt: data.optimized_character_prompt || "",
      optimized_scene_prompt: data.optimized_scene_prompt || "",
      reference_used: Boolean(data.reference_used),
      reference_strength: data.reference_strength ?? null,
    },
  };
  currentGalleryRecordId = record.id;
  if (galleryBackend === "cloud") {
    await cloudRequest("/api/gallery/records", {
      method: "POST",
      body: JSON.stringify(record),
    });
    $("save-r2").textContent = "R2に保存済み";
    return;
  }
  await storePut(HISTORY_STORE, record);
}

async function openPromptHistory(workflow) {
  historyTargetWorkflow = workflow;
  const container = $("history-grid");
  container.replaceChildren(createElement("div", "history-empty", "履歴を読み込んでいます…"));
  $("history-dialog").showModal();
  try {
    const cloudData = galleryBackend === "cloud"
      ? await cloudRequest("/api/gallery/records")
      : null;
    const records = (cloudData ? cloudData.items : await storeGetAll(HISTORY_STORE))
      .sort((a, b) => b.createdAt - a.createdAt);
    container.replaceChildren();
    if (!records.length) {
      container.append(createElement("div", "history-empty", "生成履歴はまだありません。"));
      return;
    }
    records.forEach((record) => {
      const card = createElement("button", "history-card");
      card.type = "button";
      const image = document.createElement("img");
      image.src = galleryImageSource(record);
      image.alt = record.workflow === "character" ? "キャラクター生成履歴" : "一貫再生成履歴";
      const copy = createElement("div");
      copy.append(
        createElement("strong", "", `${record.workflow === "character" ? "キャラクター" : "背景・修正"} / ${new Date(record.createdAt).toLocaleString("ja-JP")}`),
        createElement("span", "", record.prompt),
      );
      card.append(image, copy);
      card.addEventListener("click", () => applyHistoryPrompt(record));
      container.append(card);
    });
  } catch (error) {
    container.replaceChildren(createElement("div", "history-empty", `履歴を読み込めませんでした: ${error.message}`));
  }
}

function applyHistoryPrompt(record) {
  const textarea = $(promptId(historyTargetWorkflow));
  const hasCurrentPrompt = Boolean(
    textarea.value.trim() || selectedCatalog[historyTargetWorkflow].length
  );
  if (hasCurrentPrompt && !window.confirm("現在入力・選択されているプロンプトを履歴の内容で上書きします。よろしいですか？")) return;
  textarea.value = record.prompt;
  selectedCatalog[historyTargetWorkflow] = [];
  renderSelected(historyTargetWorkflow);
  const generationSettings = record.metadata?.generation_settings || {};
  applyStyleSettings(
    record.metadata?.style_settings
      || generationSettings.style
      || generationSettings.style_settings,
  );
  $("history-dialog").close();
  textarea.focus();
}

function galleryRecordFolder(record) {
  return record.folderId || "";
}

function renderGalleryFolders() {
  const container = $("gallery-folders");
  container.replaceChildren();
  const folderEntries = [
    {id: "", name: "トップ", count: galleryRecords.filter((record) => !galleryRecordFolder(record)).length},
    ...galleryFolders.map((folder) => ({
      ...folder,
      count: galleryRecords.filter((record) => galleryRecordFolder(record) === folder.id).length,
    })),
  ];
  folderEntries.forEach((folder) => {
    const button = createElement("button", "folder-item");
    button.type = "button";
    button.setAttribute("aria-pressed", String(folder.id === activeGalleryFolderId));
    button.append(
      createElement("span", "", folder.name),
      createElement("span", "folder-item-count", String(folder.count)),
    );
    button.addEventListener("click", () => {
      activeGalleryFolderId = folder.id;
      renderGallery();
    });
    container.append(button);
  });
  const isCustomFolder = Boolean(activeGalleryFolderId);
  $("gallery-folder-rename").disabled = !isCustomFolder;
  $("gallery-folder-delete").disabled = !isCustomFolder;
}

function galleryFolderName(folderId) {
  return folderId
    ? (galleryFolders.find((folder) => folder.id === folderId)?.name || "フォルダ")
    : "トップ";
}

function openGalleryPreview(record) {
  currentGalleryRecordId = record.id;
  $("gallery-preview-image").src = galleryImageSource(record);
  $("gallery-preview-title").textContent = record.workflow === "character" ? "キャラクター生成" : "一貫再生成 / 修正";
  $("gallery-preview-meta").textContent = `${new Date(record.createdAt).toLocaleString("ja-JP")} / ${galleryFolderName(galleryRecordFolder(record))}`;
  $("gallery-preview-prompt").textContent = record.prompt || record.optimizedPrompt || "";
  $("gallery-preview-dialog").showModal();
}

function closeGalleryContextMenu() {
  $("gallery-context-menu").hidden = true;
  galleryContextRecordId = "";
}

function openGalleryContextMenu(record, clientX, clientY) {
  galleryContextRecordId = record.id;
  const menu = $("gallery-context-menu");
  menu.hidden = false;
  const width = 250;
  const height = 132;
  menu.style.left = `${Math.max(8, Math.min(clientX, window.innerWidth - width - 8))}px`;
  menu.style.top = `${Math.max(8, Math.min(clientY, window.innerHeight - height - 8))}px`;
}

async function useGalleryRecordForRegeneration(record) {
  if (!record) return;
  closeGalleryContextMenu();
  if ($("gallery-preview-dialog").open) $("gallery-preview-dialog").close();
  try {
    const trainingImage = await galleryRecordAsTrainingImage(record);
    const image = trainingImage.image.slice(trainingImage.image.indexOf(",") + 1);
    const metadata = record.metadata || {};
    const characterPrompt = metadata.character_prompt
      || metadata.original_character_prompt
      || metadata.optimized_character_prompt
      || record.prompt
      || "";
    setComposeSource(image, "ギャラリーの参照画像");
    rootCharacterPrompt = metadata.character_prompt || characterPrompt;
    rootOptimizedCharacterPrompt = metadata.optimized_character_prompt || characterPrompt;
    rootScenePrompt = metadata.scene_prompt || "";
    rootOptimizedScenePrompt = metadata.optimized_scene_prompt || rootScenePrompt;
    rootPrompt = metadata.original_prompt || record.prompt || characterPrompt;
    rootOptimizedPrompt = metadata.optimized_prompt || record.optimizedPrompt || characterPrompt;
    $("compose-character-definition").value = rootCharacterPrompt || rootOptimizedCharacterPrompt;
    $("compose-method").value = "regenerate";
    applyLoraMetadataToWorkflow(metadata, "compose");
    syncComposeEditControls();
    setWorkflow("compose");
  } catch (error) {
    setError(`参照画像を読み込めませんでした: ${error.message}`);
  }
}

function openGalleryMove(record) {
  closeGalleryContextMenu();
  movingGalleryRecordId = record.id;
  const target = $("gallery-move-target");
  target.replaceChildren(new Option("トップ", ""));
  galleryFolders.forEach((folder) => target.add(new Option(folder.name, folder.id)));
  target.value = galleryRecordFolder(record);
  if (!$("gallery-move-dialog").open) $("gallery-move-dialog").showModal();
}

function renderGallery() {
  renderGalleryFolders();
  const records = galleryRecords
    .filter((record) => galleryRecordFolder(record) === activeGalleryFolderId)
    .sort((a, b) => b.createdAt - a.createdAt);
  $("gallery-title").textContent = galleryFolderName(activeGalleryFolderId);
  $("gallery-count").textContent = `${records.length}件`;
  const container = $("gallery-grid");
  container.replaceChildren();
  if (!records.length) {
    container.append(createElement("div", "gallery-empty", "このフォルダには画像がありません。"));
    return;
  }
  records.forEach((record) => {
    const card = createElement("article", "gallery-card");
    const thumbnail = createElement("button", "gallery-thumbnail");
    thumbnail.type = "button";
    const image = document.createElement("img");
    image.src = galleryImageSource(record);
    image.alt = record.workflow === "character" ? "生成したキャラクター" : "一貫再生成画像";
    thumbnail.append(image);
    let longPressTimer = null;
    let longPressed = false;
    thumbnail.addEventListener("pointerdown", () => {
      longPressed = false;
      longPressTimer = window.setTimeout(() => {
        longPressed = true;
        const bounds = thumbnail.getBoundingClientRect();
        openGalleryContextMenu(record, bounds.left + 16, bounds.top + 16);
      }, 650);
    });
    ["pointerup", "pointercancel", "pointerleave"].forEach((eventName) => {
      thumbnail.addEventListener(eventName, () => window.clearTimeout(longPressTimer));
    });
    thumbnail.addEventListener("click", () => {
      if (longPressed) {
        longPressed = false;
        return;
      }
      openGalleryPreview(record);
    });
    thumbnail.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      window.clearTimeout(longPressTimer);
      openGalleryContextMenu(record, event.clientX, event.clientY);
    });
    const copy = createElement("div", "gallery-card-copy");
    copy.append(
      createElement("strong", "", record.workflow === "character" ? "キャラクター生成" : "一貫再生成 / 修正"),
      createElement("span", "", new Date(record.createdAt).toLocaleString("ja-JP")),
    );
    const menu = createElement("button", "gallery-card-menu", "⋮");
    menu.type = "button";
    menu.title = "画像メニュー";
    menu.setAttribute("aria-label", "画像メニュー");
    menu.addEventListener("click", (event) => {
      const bounds = event.currentTarget.getBoundingClientRect();
      openGalleryContextMenu(record, bounds.right - 240, bounds.bottom + 4);
    });
    card.append(thumbnail, copy, menu);
    container.append(card);
  });
}

async function loadGallery() {
  try {
    if (galleryBackend === "cloud") {
      const [records, folders] = await Promise.all([
        cloudRequest("/api/gallery/records"),
        cloudRequest("/api/gallery/folders"),
      ]);
      galleryRecords = records.items || [];
      galleryFolders = folders.items || [];
    } else {
      [galleryRecords, galleryFolders] = await Promise.all([
        storeGetAll(HISTORY_STORE),
        storeGetAll(FOLDERS_STORE),
      ]);
    }
    galleryFolders.sort((a, b) => a.createdAt - b.createdAt);
    if (activeGalleryFolderId && !galleryFolders.some((folder) => folder.id === activeGalleryFolderId)) {
      activeGalleryFolderId = "";
    }
    renderGallery();
  } catch (error) {
    setError(`ギャラリーを読み込めませんでした: ${error.message}`);
  }
}

async function createGalleryFolder() {
  const input = $("gallery-folder-name");
  const name = input.value.trim();
  if (!name) return setError("フォルダ名を入力してください。");
  const folder = {id: crypto.randomUUID(), name, createdAt: Date.now(), updatedAt: Date.now()};
  if (galleryBackend === "cloud") {
    await cloudRequest("/api/gallery/folders", {method: "POST", body: JSON.stringify(folder)});
  } else {
    await storePut(FOLDERS_STORE, folder);
  }
  galleryFolders.push(folder);
  activeGalleryFolderId = folder.id;
  input.value = "";
  renderGallery();
  setError();
}

async function renameGalleryFolder() {
  const folder = galleryFolders.find((entry) => entry.id === activeGalleryFolderId);
  if (!folder) return;
  const name = window.prompt("フォルダの新しい名前", folder.name)?.trim();
  if (!name || name === folder.name) return;
  folder.name = name;
  folder.updatedAt = Date.now();
  if (galleryBackend === "cloud") {
    await cloudRequest(`/api/gallery/folders/${encodeURIComponent(folder.id)}`, {
      method: "PATCH",
      body: JSON.stringify(folder),
    });
  } else {
    await storePut(FOLDERS_STORE, folder);
  }
  renderGallery();
}

async function deleteGalleryFolder() {
  const folder = galleryFolders.find((entry) => entry.id === activeGalleryFolderId);
  if (!folder || !window.confirm(`「${folder.name}」を削除しますか？ 中の画像はトップへ戻ります。`)) return;
  const recordsToMove = galleryRecords.filter((record) => galleryRecordFolder(record) === folder.id);
  if (galleryBackend === "cloud") {
    await Promise.all(recordsToMove.map((record) => cloudRequest(
      `/api/gallery/records/${encodeURIComponent(record.id)}`,
      {method: "PATCH", body: JSON.stringify({folderId: ""})},
    )));
    await cloudRequest(`/api/gallery/folders/${encodeURIComponent(folder.id)}`, {method: "DELETE"});
  } else {
    await Promise.all(recordsToMove.map((record) => {
      record.folderId = "";
      return storePut(HISTORY_STORE, record);
    }));
    await storeDelete(FOLDERS_STORE, folder.id);
  }
  galleryFolders = galleryFolders.filter((entry) => entry.id !== folder.id);
  activeGalleryFolderId = "";
  renderGallery();
}

async function moveGalleryRecord() {
  const record = galleryRecords.find((entry) => entry.id === movingGalleryRecordId);
  if (!record) return;
  record.folderId = $("gallery-move-target").value;
  if (galleryBackend === "cloud") {
    await cloudRequest(`/api/gallery/records/${encodeURIComponent(record.id)}`, {
      method: "PATCH",
      body: JSON.stringify({folderId: record.folderId}),
    });
  } else {
    await storePut(HISTORY_STORE, record);
  }
  $("gallery-move-dialog").close();
  movingGalleryRecordId = "";
  renderGallery();
}

function syncLoraTrainingCategory() {
  const category = $("lora-category").value;
  const categoryLabels = {
    character: "キャラクター",
    style: "画風",
    pose: "ポーズ",
    background: "背景",
  };
  const descriptions = {
    character: "同じ人物の角度・表情・ポーズ・衣装が異なる画像を使い、画風は別LoRAへ分離します。",
    style: "同じ画風で、人物・髪・衣装・背景・構図が異なる画像を使います。同一人物だけのデータは避けます。",
    pose: "異なる人物で同じ骨格・ポーズを示す画像を使用します。",
    background: "人物を固定せず、同じ背景表現を多様な構図・時間帯で用意します。",
  };
  const recommended = loraRecommendedCounts[category] || 10;
  const maximum = loraImageLimits[category] || 30;
  $("lora-training-title").textContent = `${categoryLabels[category]}LoRAを追加学習`;
  $("lora-training-description").textContent = `${descriptions[category]} ${recommended}枚以上を推奨します。`;
  $("lora-upload-help").textContent = `${categoryLabels[category]}LoRAは最大${maximum}枚まで選択できます。`;
  $("lora-identity-field").hidden = category !== "character";
  $("lora-identity-negative-field").hidden = category !== "character";
  if (category !== "character") {
    $("lora-identity-prompt").value = "";
    $("lora-identity-negative-prompt").value = "";
  }
  updateLoraTrainingQuality();
}

function updateLoraTrainingQuality() {
  const category = $("lora-category").value;
  const count = loraTrainingSelectedIds.size + loraTrainingUploads.length;
  const recommended = loraRecommendedCounts[category] || 10;
  const maximum = loraImageLimits[category] || 30;
  const categoryLabel = {
    character: "Character",
    style: "Style",
    pose: "Pose",
    background: "Background",
  }[category];
  const quality = $("lora-training-quality");
  if (count > maximum) {
    quality.classList.remove("is-ready");
    quality.textContent = `${count}枚選択済み。${categoryLabel} LoRAの上限${maximum}枚を超えています。`;
  } else if (count >= recommended) {
    quality.classList.add("is-ready");
    quality.textContent = `${count}枚選択済み。${categoryLabel} LoRAの推奨枚数を満たしています。`;
  } else {
    quality.classList.remove("is-ready");
    quality.textContent = count
      ? `${count}枚選択済み。学習は開始できますが、${recommended}枚未満は試験品質です。`
      : `学習画像を1枚以上選択してください。${recommended}枚以上を推奨します。`;
  }
}

function renderLoraDataset() {
  const grid = $("lora-dataset-grid");
  grid.replaceChildren();
  const records = [...galleryRecords].sort((a, b) => b.createdAt - a.createdAt);
  records.forEach((record) => {
    const label = createElement("label", "lora-dataset-item");
    const image = document.createElement("img");
    image.src = galleryImageSource(record);
    image.alt = "LoRA学習候補";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = loraTrainingSelectedIds.has(record.id);
    label.classList.toggle("is-selected", checkbox.checked);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) loraTrainingSelectedIds.add(record.id);
      else loraTrainingSelectedIds.delete(record.id);
      label.classList.toggle("is-selected", checkbox.checked);
      updateLoraTrainingQuality();
    });
    label.append(image, checkbox);
    grid.append(label);
  });
  loraTrainingUploads.forEach((upload) => {
    const label = createElement("label", "lora-dataset-item lora-upload-item is-selected");
    const image = document.createElement("img");
    image.src = upload.dataUrl;
    image.alt = upload.name;
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.addEventListener("change", () => {
      if (!checkbox.checked) {
        loraTrainingUploads = loraTrainingUploads.filter((item) => item.id !== upload.id);
        renderLoraDataset();
        updateLoraTrainingQuality();
      }
    });
    label.append(image, checkbox);
    grid.append(label);
  });
  if (!records.length && !loraTrainingUploads.length) {
    grid.append(createElement("div", "history-empty", "ギャラリー画像または端末の画像を追加してください。"));
  }
  updateLoraTrainingQuality();
}

function openLoraTraining(record) {
  closeGalleryContextMenu();
  if ($("gallery-preview-dialog").open) $("gallery-preview-dialog").close();
  loraTrainingSelectedIds = new Set(record ? [record.id] : []);
  loraTrainingUploads = [];
  $("lora-name").value = "";
  $("lora-trigger-word").value = "";
  $("lora-identity-prompt").value = "";
  $("lora-identity-negative-prompt").value = "";
  $("lora-category").value = "character";
  $("lora-steps").value = "0";
  $("lora-upload").value = "";
  $("lora-training-progress").hidden = true;
  $("lora-training-progress-fill").style.width = "0%";
  $("lora-training-start").disabled = false;
  $("lora-training-start").textContent = "追加学習を開始";
  syncLoraTrainingCategory();
  renderLoraDataset();
  if (!$("lora-training-dialog").open) $("lora-training-dialog").showModal();
}

async function galleryRecordAsTrainingImage(record) {
  const source = galleryImageSource(record);
  if (!source) throw new Error("選択画像のデータを取得できません。");
  let dataUrl = source;
  if (!source.startsWith("data:")) {
    const response = await fetch(source);
    if (!response.ok) throw new Error("ギャラリー画像を学習用に取得できません。");
    dataUrl = await fileAsDataUrl(await response.blob());
  }
  return {
    image: dataUrl,
    source_id: record.id,
    prompt: record.prompt || record.optimizedPrompt || "",
    caption: record.optimizedPrompt || record.metadata?.optimized_prompt || record.prompt || "",
  };
}

async function startLoraTraining() {
  const name = $("lora-name").value.trim();
  const triggerWord = $("lora-trigger-word").value.trim();
  const identityPrompt = $("lora-identity-prompt").value.trim();
  const identityNegativePrompt = $("lora-identity-negative-prompt").value.trim();
  const category = $("lora-category").value;
  const selectedRecords = galleryRecords.filter((record) => loraTrainingSelectedIds.has(record.id));
  const imageCount = selectedRecords.length + loraTrainingUploads.length;
  const maximum = loraImageLimits[category] || 30;
  if (!name) return setError("LoRA名を入力してください。");
  if (!/^[A-Za-z][A-Za-z0-9_-]{2,63}$/.test(triggerWord)) {
    return setError("トリガーワードは英字で始まる3〜64文字の英数字、_、-で入力してください。");
  }
  if (!imageCount) return setError("学習画像を1枚以上選択してください。");
  if (imageCount > maximum) return setError(`学習画像は最大${maximum}枚です。`);

  const button = $("lora-training-start");
  button.disabled = true;
  button.textContent = "画像を準備中";
  setError();
  $("lora-training-progress").hidden = false;
  $("lora-training-progress-title").textContent = "学習画像を準備中";
  $("lora-training-progress-message").textContent = `${imageCount}枚を読み込んでいます`;
  try {
    const galleryImages = await Promise.all(selectedRecords.map(galleryRecordAsTrainingImage));
    const uploadImages = loraTrainingUploads.map((upload) => ({
      image: upload.dataUrl,
      source_id: upload.id,
      prompt: "",
    }));
    const start = await backendRequest("/api/lora/train/start", {
      method: "POST",
      body: JSON.stringify({
        name,
        trigger_word: triggerWord,
        identity_prompt: identityPrompt,
        identity_negative_prompt: identityNegativePrompt,
        category,
        model_type: $("lora-model-type").value,
        steps: Number($("lora-steps").value),
        images: [...galleryImages, ...uploadImages],
      }),
    });
    button.textContent = "追加学習中";
    await loadLoraModels();
    await new Promise((resolve, reject) => {
      const stream = new EventSource(`/api/lora/train/stream/${start.job_id}`);
      stream.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "status") {
          $("lora-training-progress-title").textContent = "LoRAを学習中";
          $("lora-training-progress-message").textContent = data.message || "";
        } else if (data.type === "progress") {
          const percent = Math.round((data.step / data.total) * 100);
          $("lora-training-progress-title").textContent = `LoRAを学習中 ${percent}%`;
          $("lora-training-progress-message").textContent = `${data.step} / ${data.total} steps`;
          $("lora-training-progress-fill").style.width = `${percent}%`;
        } else if (data.type === "done") {
          stream.close();
          resolve(data.model);
        } else if (data.type === "error") {
          stream.close();
          reject(new Error(data.message || "LoRA学習に失敗しました。"));
        }
      };
      stream.onerror = () => {
        stream.close();
        reject(new Error("LoRA学習の進捗接続が切れました。"));
      };
    }).then(async (model) => {
      $("lora-training-progress-title").textContent = "LoRA学習が完了しました";
      $("lora-training-progress-message").textContent = `${model.name} を生成画面で使用できます。`;
      $("lora-training-progress-fill").style.width = "100%";
      button.textContent = "学習完了";
      await loadLoraModels(model.id);
    });
  } catch (error) {
    setError(error.message);
    $("lora-training-progress-title").textContent = "LoRA学習に失敗しました";
    $("lora-training-progress-message").textContent = error.message;
    button.disabled = false;
    button.textContent = "もう一度開始";
  }
}

function fileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function setComposeSource(image, label) {
  composeSourceImage = image || "";
  composeSourceLabel = label || "";
  const hasSource = Boolean(composeSourceImage);
  $("compose-source-preview").hidden = !hasSource;
  $("compose-source-empty").hidden = hasSource;
  if (hasSource) {
    $("compose-source-image").src = `data:image/png;base64,${composeSourceImage}`;
    $("compose-source-image").alt = composeSourceLabel || "編集元の画像";
  }
}

function addEditHistory(instruction) {
  $("edit-history").append(createElement("div", "user", `あなた: ${instruction}`));
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
    const batchImages = [];
    const stream = new EventSource(`/api/generate/stream/${start.job_id}`);
    stream.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "status") {
        const imageLabel = data.image_total > 1
          ? `${data.image_index} / ${data.image_total} 枚目 `
          : "";
        $("progress-title").textContent = data.phase === "generate"
          ? `${imageLabel}画像を生成中`
          : "プロンプトを準備中";
        $("progress-message").textContent = data.message || "";
      } else if (data.type === "optimized_prompt") {
        $("progress-title").textContent = "画像を生成中";
        $("progress-message").textContent = "モデルへプロンプトを送信しました";
      } else if (data.type === "progress") {
        const percent = Math.round((data.step / data.total) * 100);
        const imageLabel = data.image_total > 1
          ? `${data.image_index} / ${data.image_total} 枚目 `
          : "";
        $("progress-title").textContent = `${imageLabel}画像を生成中 ${percent}%`;
        $("progress-message").textContent = `${data.step} / ${data.total} steps`;
        $("progress-fill").style.width = `${percent}%`;
      } else if (data.type === "batch_image") {
        batchImages.push(data);
        $("progress-title").textContent = `${data.image_index} / ${data.image_total} 枚目が完了`;
        $("progress-message").textContent = "次の画像を同じ設定で順番に生成します";
        $("progress-fill").style.width = `${Math.round((data.image_index / data.image_total) * 100)}%`;
      } else if (data.type === "done") {
        stream.close();
        data.images = batchImages.length
          ? batchImages
          : [{
            image: data.image,
            settings: data.settings,
            image_index: 1,
            image_total: 1,
          }];
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

function updateLoraState(data) {
  if (data.workflow === "character") {
    rootLoraMetadata = (
      data.character_lora_id || data.lora_id || data.style_lora_id
    ) ? loraMetadataFromResult(data) : {};
  } else if (
    data.workflow === "compose"
    && !data.character_lora_id
    && !data.lora_id
    && !data.style_lora_id
    && (rootLoraMetadata.character_lora_id || rootLoraMetadata.style_lora_id)
  ) {
    Object.assign(data, rootLoraMetadata);
  } else if (
    data.workflow === "compose"
    && (data.character_lora_id || data.lora_id || data.style_lora_id)
  ) {
    rootLoraMetadata = loraMetadataFromResult(data);
  }
}

function showResult(data, historyPrompt) {
  updateLoraState(data);
  currentImage = data.image;
  currentGalleryRecordId = "";
  currentImages[data.workflow] = data.image;
  currentMetadata = {
    original_prompt: rootPrompt || data.original_prompt,
    free_prompt: data.free_prompt || "",
    catalog_prompt: data.catalog_prompt || "",
    edit_history: [...editInstructions],
    optimized_prompt: data.optimized_prompt,
    optimizer_source: data.optimizer_source,
    intent_notes: data.intent_notes,
    refine_enabled: data.refine_enabled,
    generation_settings: data.settings,
    style_settings: data.settings?.style || styleSettings(),
    image_model_profile: data.image_model_profile || "",
    image_model_label: data.image_model_label || "",
    workflow: data.workflow,
    editor_model: data.editor_model || "",
    edit_strength: data.edit_strength ?? null,
    edit_scope: data.edit_scope || "",
    background_mask: data.background_mask || "",
    lora_id: data.lora_id || "",
    lora_name: data.lora_name || "",
    lora_trigger_word: data.lora_trigger_word || "",
    lora_weight: data.lora_weight ?? null,
    character_lora_id: data.character_lora_id || data.lora_id || "",
    character_lora_name: data.character_lora_name || data.lora_name || "",
    character_lora_trigger_word: data.character_lora_trigger_word || data.lora_trigger_word || "",
    character_lora_weight: data.character_lora_weight ?? data.lora_weight ?? null,
    style_lora_id: data.style_lora_id || "",
    style_lora_name: data.style_lora_name || "",
    style_lora_trigger_word: data.style_lora_trigger_word || "",
    style_lora_weight: data.style_lora_weight ?? null,
    applied_loras: data.applied_loras || [],
    generation_intent: data.generation_intent || "",
    character_prompt: data.character_prompt || rootCharacterPrompt || "",
    scene_prompt: data.scene_prompt || rootScenePrompt || "",
    optimized_character_prompt: data.optimized_character_prompt || rootOptimizedCharacterPrompt || "",
    optimized_scene_prompt: data.optimized_scene_prompt || "",
    reference_used: Boolean(data.reference_used),
    reference_strength: data.reference_strength ?? null,
  };
  $("result-image").src = `data:image/png;base64,${data.image}`;
  $("optimized-prompt").textContent = data.optimized_prompt;
  $("result-workflow-label").textContent = data.generation_intent === "story_illustration"
    ? "ストーリー用一枚絵"
    : data.generation_intent === "consistent_regeneration"
      ? "キャラクター一貫再生成"
      : data.workflow === "character" ? "キャラクター生成" : "局所修正";
  $("use-result-in-compose").hidden = data.workflow !== "character";
  $("save-r2").textContent = "R2に保存";
  if (data.workflow === "character") {
    setComposeSource(data.image, "キャラクター生成の結果");
  }
  if (data.workflow === "compose") {
    setComposeSource(data.image, "直前の再生成結果");
  }
  finishProgress();
  savePromptHistory(data, historyPrompt).catch((error) => console.warn("Prompt history could not be saved", error));
}

function selectBatchResult(index) {
  const entry = batchResults[index];
  if (!entry) return;
  selectedBatchIndex = index;
  currentImage = entry.data.image;
  currentMetadata = entry.metadata;
  currentGalleryRecordId = "";
  currentImages[entry.data.workflow] = entry.data.image;
  $("result-image").src = "data:image/png;base64," + entry.data.image;
  $("optimized-prompt").textContent = entry.data.optimized_prompt;
  $("result-workflow-label").textContent = entry.label;
  $("use-result-in-compose").hidden = entry.data.workflow !== "character";
  $("save-r2").textContent = "R2に保存";
  $("result-batch-grid").querySelectorAll(".batch-result-card").forEach(
    (card, cardIndex) => card.setAttribute(
      "aria-pressed",
      String(cardIndex === selectedBatchIndex),
    ),
  );
  if (entry.data.workflow === "character") {
    setComposeSource(entry.data.image, "キャラクター生成の結果");
  }
  if (entry.data.workflow === "compose") {
    setComposeSource(entry.data.image, "直前の再生成結果");
  }
}

function renderBatchResults() {
  const grid = $("result-batch-grid");
  const multiple = batchResults.length > 1;
  grid.replaceChildren();
  grid.hidden = !multiple;
  $("result-single-stage").hidden = multiple;
  $("batch-result-summary").hidden = !multiple;
  if (!multiple) return;
  $("batch-result-summary").textContent = (
    batchResults.length
    + "枚を1枚ずつ生成しました。画像を選ぶと、保存・再生成の対象が切り替わります。"
  );
  batchResults.forEach((entry, index) => {
    const card = createElement("button", "batch-result-card");
    card.type = "button";
    card.setAttribute("aria-pressed", String(index === selectedBatchIndex));
    const image = document.createElement("img");
    image.src = "data:image/png;base64," + entry.data.image;
    image.alt = "生成画像 " + (index + 1);
    const seed = entry.data.settings?.seed;
    const caption = createElement(
      "span",
      "",
      (index + 1) + " / " + batchResults.length
        + (seed == null ? "" : " ・ Seed " + seed),
    );
    card.append(image, caption);
    card.addEventListener("click", () => selectBatchResult(index));
    grid.append(card);
  });
}

function showGenerationResults(data, historyPrompt) {
  const images = data.images || [{
    image: data.image,
    settings: data.settings,
    image_index: 1,
    image_total: 1,
  }];
  batchResults = [];
  images.forEach((imageData) => {
    const resultData = {
      ...data,
      image: imageData.image,
      settings: imageData.settings || data.settings,
      batch_index: imageData.image_index || 1,
      batch_total: imageData.image_total || images.length,
    };
    delete resultData.images;
    showResult(resultData, historyPrompt);
    batchResults.push({
      data: resultData,
      metadata: {...currentMetadata},
      label: $("result-workflow-label").textContent,
    });
  });
  selectedBatchIndex = 0;
  renderBatchResults();
  selectBatchResult(0);
}

async function generateCharacter() {
  const promptParts = promptPartsForWorkflow("character");
  const prompt = promptParts.prompt;
  const scenePrompt = $("character-scene-prompt").value.trim();
  const includeScene = Boolean(scenePrompt && $("character-auto-background").checked);
  if (!prompt) return setError("キャラクターの要素を入力するか、カタログから選択してください。");
  const button = $("generate-character");
  button.disabled = true;
  beginProgress("キャラクター設定を確認しています");
  try {
    const data = await runJob({
      workflow: "character", mode: "t2i", prompt,
      character_prompt: prompt,
      scene_prompt: includeScene ? scenePrompt : "",
      generation_intent: includeScene ? "story_illustration" : "character_asset",
      free_prompt: promptParts.freeText,
      catalog_prompt: promptParts.catalogPrompt,
      refine_enabled: $("character-refine-enabled").checked,
      batch_count: batchCountForWorkflow("character"),
      ...loraGenerationSettings("character"),
      ...generationSettings(),
    });
    rootCharacterPrompt = prompt;
    rootOptimizedCharacterPrompt = data.optimized_character_prompt || data.optimized_prompt || prompt;
    rootScenePrompt = includeScene ? scenePrompt : "";
    rootOptimizedScenePrompt = data.optimized_scene_prompt || rootScenePrompt;
    rootPrompt = includeScene ? `人物: ${prompt}\nシーン: ${scenePrompt}` : prompt;
    rootOptimizedPrompt = data.optimized_prompt || prompt;
    editInstructions = [];
    $("edit-history").replaceChildren();
    $("compose-character-definition").value = prompt;
    applyLoraMetadataToWorkflow(loraMetadataFromResult(data), "compose");
    showGenerationResults(data, rootPrompt);
  } catch (error) {
    setError(error.message);
    $("progress").hidden = true;
    $("empty").hidden = Boolean(currentImage);
    $("result").hidden = !currentImage;
  } finally {
    button.disabled = false;
  }
}

async function generateCompose() {
  const promptParts = promptPartsForWorkflow("compose");
  const prompt = promptParts.prompt;
  const method = $("compose-method").value;
  const characterDefinition = $("compose-character-definition").value.trim()
    || rootCharacterPrompt
    || rootOptimizedCharacterPrompt;
  if (!composeSourceImage) return setError("キャラクター参照画像を選択してください。");
  if (!prompt) return setError("背景・構図・修正内容を入力するか、カタログから選択してください。");
  if (method === "regenerate" && !characterDefinition) {
    return setError("保持するキャラクター定義を入力してください。LoRAだけでなく、髪・目・衣装などの固定条件も使用します。");
  }
  const button = $("generate-compose");
  button.disabled = true;
  beginProgress(method === "regenerate"
    ? "キャラクター条件と変更内容を分けて準備しています"
    : "手動マスクの局所修正を準備しています");
  addEditHistory(prompt);
  try {
    const maskFile = $("compose-mask").files[0];
    const maskImage = maskFile ? await fileAsDataUrl(maskFile) : "";
    if (method === "manual_inpaint" && !maskImage) {
      throw new Error("局所修正では、白い領域を変更対象にした手動マスクを選択してください。");
    }
    const lockCharacter = method === "regenerate" && $("compose-lock-character").checked;
    const payload = {
      workflow: "compose",
      mode: method === "regenerate" ? "t2i" : "edit",
      generation_intent: method === "regenerate" ? "consistent_regeneration" : "manual_edit",
      prompt,
      free_prompt: promptParts.freeText,
      catalog_prompt: promptParts.catalogPrompt,
      refine_enabled: $("compose-refine-enabled").checked,
      batch_count: method === "regenerate" ? batchCountForWorkflow("compose") : 1,
      ...generationSettings(),
    };
    if (method === "regenerate") {
      Object.assign(payload, loraGenerationSettings("compose"), {
        character_prompt: characterDefinition,
        lock_character_outfit: lockCharacter,
        locked_character_prompt: rootOptimizedCharacterPrompt || characterDefinition,
        source_scene_prompt: rootOptimizedScenePrompt || rootScenePrompt,
        reference_image: composeSourceImage,
        reference_strength: Number($("compose-reference-strength").value) / 100,
      });
    } else {
      Object.assign(payload, {
        source_image: composeSourceImage,
        mask_image: maskImage,
        editor_model: "waifu_inpaint_xl",
        edit_scope: "full",
        edit_strength: Number($("compose-strength").value) / 100,
      });
    }
    const data = await runJob(payload);
    rootCharacterPrompt = characterDefinition || rootCharacterPrompt;
    rootOptimizedCharacterPrompt = data.optimized_character_prompt || rootOptimizedCharacterPrompt;
    rootScenePrompt = prompt;
    rootOptimizedScenePrompt = data.optimized_scene_prompt || prompt;
    rootPrompt = rootPrompt || `人物: ${rootCharacterPrompt}\n変更: ${prompt}`;
    editInstructions.push(prompt);
    showGenerationResults(data, prompt);
    $("edit-history").append(createElement(
      "div",
      "assistant",
      method === "regenerate"
        ? "AI: LoRAと参照画像を使って一枚絵を再生成しました。"
        : "AI: 手動マスクの範囲を局所修正しました。",
    ));
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
    if (galleryBackend === "cloud") {
      const record = {
        id: currentGalleryRecordId || crypto.randomUUID(),
        createdAt: Date.now(),
        workflow: currentMetadata.workflow || "compose",
        image: currentImage,
        prompt: currentMetadata.original_prompt || "",
        optimizedPrompt: currentMetadata.optimized_prompt || "",
        folderId: "",
        metadata: currentMetadata,
      };
      await cloudRequest("/api/gallery/records", {
        method: "POST",
        body: JSON.stringify(record),
      });
      currentGalleryRecordId = record.id;
      button.textContent = "R2に保存済み";
      return;
    }
    const response = await fetch("/api/save-to-r2", {
      method: "POST", headers: {"Content-Type": "application/json"},
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

function bindEvents() {
  $("preset").addEventListener("change", () => applyPreset(true));
  styleFields.forEach(([id]) => $(id).addEventListener("input", () => {
    syncSlider(id);
    persistStyleSettings();
  }));
  $("compose-strength").addEventListener("input", () => {
    $("compose-strength-value").value = $("compose-strength").value;
  });
  $("compose-reference-strength").addEventListener("input", () => {
    $("compose-reference-strength-value").value = $("compose-reference-strength").value;
  });
  ["character", "compose"].forEach((workflow) => {
    $(`${workflow}-lora-weight`).addEventListener("input", () => {
      $(`${workflow}-lora-weight-value`).value = $(`${workflow}-lora-weight`).value;
    });
    $(`${workflow}-style-lora-weight`).addEventListener("input", () => {
      $(`${workflow}-style-lora-weight-value`).value = $(`${workflow}-style-lora-weight`).value;
    });
  });
  $("compose-method").addEventListener("change", syncComposeEditControls);
  $("character-auto-background").addEventListener("change", syncCharacterCommand);
  $("character-scene-prompt").addEventListener("input", syncCharacterCommand);
  $("character-batch-count").addEventListener("input", syncCharacterCommand);
  $("compose-batch-count").addEventListener("input", syncComposeEditControls);
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
    if (currentImages.character) {
      setComposeSource(currentImages.character, "キャラクター生成の結果");
      $("compose-character-definition").value = rootCharacterPrompt || rootOptimizedCharacterPrompt;
      applyLoraMetadataToWorkflow(rootLoraMetadata, "compose");
    }
  });
  $("use-result-in-compose").addEventListener("click", () => setWorkflow("compose"));
  $("compose-source-file").addEventListener("change", async () => {
    const file = $("compose-source-file").files[0];
    if (!file) return;
    try {
      const dataUrl = await fileAsDataUrl(file);
      setComposeSource(dataUrl.slice(dataUrl.indexOf(",") + 1), file.name);
      rootLoraMetadata = {};
      rootCharacterPrompt = "";
      rootOptimizedCharacterPrompt = "";
      rootScenePrompt = "";
      rootOptimizedScenePrompt = "";
      $("compose-character-definition").value = "";
      $("compose-method").value = "regenerate";
      syncComposeEditControls();
    } catch {
      setError("画像ファイルを読み込めませんでした。");
    }
  });
  $("character-history-open").addEventListener("click", () => openPromptHistory("character"));
  $("compose-history-open").addEventListener("click", () => openPromptHistory("compose"));
  $("history-close").addEventListener("click", () => $("history-dialog").close());
  $("favorite-group-add").addEventListener("click", createFavoriteGroup);
  $("favorite-group-name").addEventListener("keydown", (event) => {
    if (event.key === "Enter") createFavoriteGroup();
  });
  $("favorite-group-select").addEventListener("change", () => {
    activeFavoriteGroupId = $("favorite-group-select").value;
    renderFavoriteSettings();
    renderCatalog("settings", catalogViews.settings);
  });
  $("favorite-group-rename").addEventListener("click", renameFavoriteGroup);
  $("favorite-group-delete").addEventListener("click", deleteFavoriteGroup);
  $("custom-prompt-add").addEventListener("click", addCustomFavoritePrompt);
  ["character", "compose"].forEach((workflow) => {
    $(`${workflow}-favorite-import`).addEventListener("click", () => importFavoriteGroup(workflow));
    $(`${workflow}-catalog-reset`).addEventListener("click", () => resetCatalogSelection(workflow));
  });
  $("gallery-folder-add").addEventListener("click", createGalleryFolder);
  $("gallery-folder-name").addEventListener("keydown", (event) => {
    if (event.key === "Enter") createGalleryFolder();
  });
  $("gallery-folder-rename").addEventListener("click", renameGalleryFolder);
  $("gallery-folder-delete").addEventListener("click", deleteGalleryFolder);
  $("gallery-preview-close").addEventListener("click", () => $("gallery-preview-dialog").close());
  $("gallery-preview-regenerate").addEventListener("click", () => {
    useGalleryRecordForRegeneration(
      galleryRecords.find((record) => record.id === currentGalleryRecordId)
    );
  });
  $("gallery-preview-train").addEventListener("click", () => {
    openLoraTraining(galleryRecords.find((record) => record.id === currentGalleryRecordId));
  });
  $("gallery-context-regenerate").addEventListener("click", () => {
    useGalleryRecordForRegeneration(
      galleryRecords.find((record) => record.id === galleryContextRecordId)
    );
  });
  $("gallery-context-train").addEventListener("click", () => {
    openLoraTraining(galleryRecords.find((record) => record.id === galleryContextRecordId));
  });
  $("gallery-context-move").addEventListener("click", () => {
    const record = galleryRecords.find((entry) => entry.id === galleryContextRecordId);
    if (record) openGalleryMove(record);
  });
  $("gallery-move-close").addEventListener("click", () => $("gallery-move-dialog").close());
  $("gallery-move-confirm").addEventListener("click", moveGalleryRecord);
  $("lora-training-close").addEventListener("click", () => $("lora-training-dialog").close());
  $("lora-category").addEventListener("change", syncLoraTrainingCategory);
  $("lora-training-start").addEventListener("click", startLoraTraining);
  $("lora-upload").addEventListener("change", async () => {
    const category = $("lora-category").value;
    const maximum = loraImageLimits[category] || 30;
    const remaining = Math.max(0, maximum - loraTrainingSelectedIds.size - loraTrainingUploads.length);
    const files = [...$("lora-upload").files].slice(0, remaining);
    for (const file of files) {
      loraTrainingUploads.push({
        id: crypto.randomUUID(),
        name: file.name,
        dataUrl: await fileAsDataUrl(file),
      });
    }
    $("lora-upload").value = "";
    renderLoraDataset();
  });
  document.addEventListener("pointerdown", (event) => {
    if (!$("gallery-context-menu").hidden && !$("gallery-context-menu").contains(event.target)) {
      closeGalleryContextMenu();
    }
  });

  catalogWorkflows.forEach((workflow) => {
    $(catalogId(workflow, "category")).addEventListener("change", () => {
      $(catalogId(workflow, "subcategory")).value = "";
      loadCatalog(workflow);
    });
    $(catalogId(workflow, "subcategory")).addEventListener("change", () => loadCatalog(workflow));
    $(catalogId(workflow, "query")).addEventListener("input", () => queueCatalogSearch(workflow));
    $(catalogId(workflow, "more")).addEventListener("click", () => loadCatalog(workflow, true));
  });
  document.querySelectorAll(".workflow-tab").forEach((button) => {
    button.addEventListener("click", () => setWorkflow(button.dataset.workflow));
  });
}

function syncCharacterCommand() {
  const story = Boolean(
    $("character-auto-background").checked
    && $("character-scene-prompt").value.trim()
  );
  const count = batchCountForWorkflow("character");
  $("generate-character").textContent = count > 1
    ? story
      ? count + "枚の一枚絵を順番に生成"
      : count + "枚のキャラクター素材を順番に生成"
    : story
      ? "シーン込みの一枚絵を生成"
      : "キャラクター素材を生成";
}

function syncComposeEditControls() {
  const regenerate = $("compose-method").value === "regenerate";
  $("compose-character-definition-field").hidden = !regenerate;
  $("compose-character-lock").hidden = !regenerate;
  $("compose-lora-field").hidden = !regenerate;
  $("compose-reference-strength-field").hidden = !regenerate;
  $("compose-mask-settings").hidden = regenerate;
  $("compose-batch-count-field").hidden = !regenerate;
  $("compose-strength-field").hidden = regenerate;
  $("compose-strength-value").value = $("compose-strength").value;
  const count = batchCountForWorkflow("compose");
  $("generate-compose").textContent = regenerate
    ? count > 1
      ? count + "枚を順番に一貫再生成"
      : "同じキャラクターで再生成"
    : "手動マスクの範囲を修正";
}

async function initialize() {
  bindEvents();
  applyPreset(false);
  restoreCachedStyleSettings();
  syncCharacterCommand();
  syncComposeEditControls();
  renderSelected("character");
  renderSelected("compose");
  try {
    await detectGalleryBackend();
    const cloudFavorites = galleryBackend === "cloud" ? await cloudRequest("/api/favorites") : null;
    favoriteGroups = (cloudFavorites ? cloudFavorites.items : await storeGetAll(FAVORITES_STORE))
      .sort((a, b) => a.createdAt - b.createdAt);
    activeFavoriteGroupId = favoriteGroups[0]?.id || "";
  } catch (error) {
    setError(`ユーザー設定を読み込めませんでした: ${error.message}`);
  }
  renderFavoriteSettings();
  try {
    await loadLoraModels();
  } catch (error) {
    setError(`LoRA一覧を読み込めませんでした: ${error.message}`);
  }
  await Promise.all(catalogWorkflows.map((workflow) => loadCatalog(workflow)));
  setWorkflow("character");
}

initialize();
