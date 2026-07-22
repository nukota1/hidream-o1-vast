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

let activeWorkflow = "character";
let composeSourceImage = "";
let composeSourceLabel = "";
let compositionForegroundImage = "";
let compositionBackgroundMask = "";
let currentImage = "";
let currentMetadata = {};
let rootPrompt = "";
let rootOptimizedPrompt = "";
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
  }
  if (workflow === "settings") renderFavoriteSettings();
  if (workflow === "gallery") loadGallery();
}

function promptForWorkflow(workflow) {
  const freeText = $(promptId(workflow)).value.trim();
  const catalogPrompt = selectedCatalog[workflow].map((item) => item.prompt).join(", ");
  // Free text expresses the user's exact intent, so it must reach Qwen and
  // CLIP before the optional catalog tags when the prompt needs compacting.
  return [freeText, catalogPrompt].filter(Boolean).join("\n");
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
      optimized_prompt: data.optimized_prompt || "",
      optimizer_source: data.optimizer_source || "",
      intent_notes: data.intent_notes || "",
      generation_settings: data.settings || {},
      workflow: data.workflow || "",
      editor_model: data.editor_model || "",
      edit_strength: data.edit_strength ?? null,
      edit_scope: data.edit_scope || "",
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
      image.alt = record.workflow === "character" ? "キャラクター生成履歴" : "背景合成履歴";
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
  $("gallery-preview-image").src = galleryImageSource(record);
  $("gallery-preview-title").textContent = record.workflow === "character" ? "キャラクター生成" : "背景合成 / 修正";
  $("gallery-preview-meta").textContent = `${new Date(record.createdAt).toLocaleString("ja-JP")} / ${galleryFolderName(galleryRecordFolder(record))}`;
  $("gallery-preview-prompt").textContent = record.prompt || record.optimizedPrompt || "";
  $("gallery-preview-dialog").showModal();
}

function openGalleryMove(record) {
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
    image.alt = record.workflow === "character" ? "生成したキャラクター" : "生成した背景合成画像";
    thumbnail.append(image);
    let longPressTimer = null;
    let longPressed = false;
    thumbnail.addEventListener("pointerdown", () => {
      longPressed = false;
      longPressTimer = window.setTimeout(() => {
        longPressed = true;
        openGalleryMove(record);
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
      openGalleryMove(record);
    });
    const copy = createElement("div", "gallery-card-copy");
    copy.append(
      createElement("strong", "", record.workflow === "character" ? "キャラクター生成" : "背景合成 / 修正"),
      createElement("span", "", new Date(record.createdAt).toLocaleString("ja-JP")),
    );
    const menu = createElement("button", "gallery-card-menu", "⋮");
    menu.type = "button";
    menu.title = "画像を移動";
    menu.setAttribute("aria-label", "画像を移動");
    menu.addEventListener("click", () => openGalleryMove(record));
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

function showResult(data, historyPrompt) {
  currentImage = data.image;
  currentGalleryRecordId = "";
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
    setComposeSource(data.image, "直前の修正結果", {preserveForeground: Boolean(data.background_mask)});
  }
  finishProgress();
  savePromptHistory(data, historyPrompt).catch((error) => console.warn("Prompt history could not be saved", error));
}

async function generateCharacter() {
  const prompt = promptForWorkflow("character");
  if (!prompt) return setError("キャラクターの要素を入力するか、カタログから選択してください。");
  const button = $("generate-character");
  button.disabled = true;
  beginProgress("キャラクター設定を確認しています");
  try {
    const data = await runJob({
      workflow: "character", mode: "t2i", prompt,
      refine_enabled: $("character-refine-enabled").checked,
      ...generationSettings(),
    });
    rootPrompt = prompt;
    rootOptimizedPrompt = data.optimized_prompt || prompt;
    editInstructions = [];
    $("edit-history").replaceChildren();
    showResult(data, prompt);
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
  const method = $("compose-method").value;
  if (!composeSourceImage) return setError("編集元の画像を選択してください。");
  if (!prompt) return setError("背景・構図・修正内容を入力するか、カタログから選択してください。");
  if (method === "integrated" && !rootPrompt) {
    return setError("イベントCGとして再生成するには、先にこのアプリでキャラクターを生成してください。任意画像を使う場合は背景だけ置換を選択してください。");
  }
  const button = $("generate-compose");
  button.disabled = true;
  beginProgress("背景と構図の指示を準備しています");
  addEditHistory(prompt);
  try {
    const maskFile = $("compose-mask").files[0];
    const maskImage = maskFile ? await fileAsDataUrl(maskFile) : "";
    const editScope = $("compose-edit-scope").value;
    const useSceneContext = Boolean(editScope === "background" && !maskImage && compositionForegroundImage && compositionBackgroundMask);
    const requestPrompt = editScope === "background" && editInstructions.length
      ? [...editInstructions, prompt].join("\n") : prompt;
    const lockCharacter = method === "integrated" && $("compose-lock-character").checked;
    const payload = {
      workflow: "compose",
      mode: method === "integrated" ? "t2i" : "edit",
      prompt: requestPrompt,
      refine_enabled: $("compose-refine-enabled").checked,
      ...generationSettings(),
    };
    if (method === "integrated" && lockCharacter) {
      payload.lock_character_outfit = true;
      payload.locked_character_prompt = rootOptimizedPrompt || rootPrompt;
    }
    if (method === "inpaint") {
      Object.assign(payload, {
        source_image: useSceneContext ? compositionForegroundImage : composeSourceImage,
        mask_image: maskImage,
        editor_model: "waifu_inpaint_xl",
        edit_scope: editScope,
        background_mask_image: useSceneContext ? compositionBackgroundMask : "",
        edit_strength: Number($("compose-strength").value) / 100,
      });
    }
    const data = await runJob(payload);
    rootPrompt = rootPrompt || prompt;
    editInstructions.push(prompt);
    showResult(data, prompt);
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
  $("preset").addEventListener("change", applyPreset);
  styleFields.forEach(([id]) => $(id).addEventListener("input", () => syncSlider(id)));
  $("compose-strength").addEventListener("input", () => {
    $("compose-strength-value").value = $("compose-strength").value;
  });
  $("compose-edit-scope").addEventListener("change", syncComposeEditControls);
  $("compose-method").addEventListener("change", syncComposeEditControls);
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
      $("compose-method").value = "inpaint";
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
  $("gallery-move-close").addEventListener("click", () => $("gallery-move-dialog").close());
  $("gallery-move-confirm").addEventListener("click", moveGalleryRecord);

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

function syncComposeEditControls() {
  const integrated = $("compose-method").value === "integrated";
  const isBackground = $("compose-edit-scope").value === "background";
  $("compose-edit-scope-field").hidden = integrated;
  $("compose-character-lock").hidden = !integrated;
  $("compose-mask-settings").hidden = integrated;
  $("compose-strength-field").hidden = integrated || isBackground;
  $("compose-strength").value = isBackground ? 85 : 55;
  $("compose-strength-value").value = $("compose-strength").value;
  $("generate-compose").textContent = integrated ? "イベントCGを生成" : "背景を合成 / 画像を修正";
}

async function initialize() {
  bindEvents();
  applyPreset();
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
  await Promise.all(catalogWorkflows.map((workflow) => loadCatalog(workflow)));
  setWorkflow("character");
}

initialize();
