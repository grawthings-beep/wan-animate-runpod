import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const SLOT_COUNT = 10;
const IMAGE_PATTERN = /\.(png|jpe?g|webp|bmp|gif)$/i;
const PROMPT_PATTERN = /\.(txt|json)$/i;
const naturalCollator = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: "base",
});

function setWidgetValue(node, widget, value) {
  const previous = widget.value;
  widget.value = value;
  widget.callback?.(value);
  node.onWidgetChanged?.(widget.name, value, previous, widget);
}

function linkedSlotNodes(selector) {
  return Array.from({ length: SLOT_COUNT }, (_, index) => {
    const input = selector.inputs?.find(
      (item) => item.name === `slot_${String(index + 1).padStart(2, "0")}`,
    );
    const link = input?.link == null ? null : selector.graph?.links?.[input.link];
    return link == null ? null : selector.graph?.getNodeById(link.origin_id);
  });
}

function slotNodesOrThrow(selector) {
  const nodes = linkedSlotNodes(selector);
  const missing = nodes
    .map((node, index) => (!node || node.type !== "WanLoopQueueSlot" ? index + 1 : null))
    .filter(Boolean);
  if (missing.length) {
    throw new Error(`slot node is not connected: ${missing.join(", ")}`);
  }
  return nodes;
}

function validateSlots(selector) {
  let nodes;
  try {
    nodes = slotNodesOrThrow(selector);
  } catch (error) {
    return [error.message];
  }
  const problems = [];
  nodes.forEach((node, index) => {
    const image = node.widgets?.find((widget) => widget.name === "image")?.value;
    const prompt = node.widgets?.find(
      (widget) => widget.name === "positive_prompt",
    )?.value;
    if (!image || image === "DigitalPastelLogo.png") {
      problems.push(`slot ${index + 1}: upload/select an image`);
    }
    if (!String(prompt ?? "").trim()) {
      problems.push(`slot ${index + 1}: positive prompt is empty`);
    }
  });
  return problems;
}

function newBatchId() {
  const timestamp = new Date().toISOString().replace(/[-:.]/g, "");
  const random = globalThis.crypto?.randomUUID?.().slice(0, 8)
    ?? Math.random().toString(16).slice(2, 10);
  return `loop10-${timestamp}-${random}`;
}

function serverImageName(response) {
  if (!response?.name) throw new Error("ComfyUI did not return an uploaded image name");
  return response.subfolder
    ? `${String(response.subfolder).replace(/\\/g, "/")}/${response.name}`
    : response.name;
}

function safeUploadName(displayName, index) {
  const leaf = String(displayName).replace(/\\/g, "/").split("/").pop();
  const cleaned = leaf.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[-.]+/, "");
  return `slot-${String(index + 1).padStart(2, "0")}-${cleaned || "image.png"}`;
}

async function uploadImage(file, displayName, index, subfolder) {
  const body = new FormData();
  body.append("image", file, safeUploadName(displayName, index));
  body.append("type", "input");
  body.append("subfolder", subfolder);
  body.append("overwrite", "true");
  const response = await api.fetchApi("/upload/image", { method: "POST", body });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.error ?? `image upload failed (${response.status})`);
  return { name: serverImageName(payload), displayName };
}

async function uploadTenImages(entries, setStatus) {
  const ordered = [...entries].sort((left, right) =>
    naturalCollator.compare(left.relativePath, right.relativePath));
  if (ordered.length !== SLOT_COUNT) {
    throw new Error(`exactly 10 images are required; found ${ordered.length}`);
  }
  const token = globalThis.crypto?.randomUUID?.().replaceAll("-", "")
    ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const subfolder = `wan-loop-imports/browser-${token}`;
  const output = Array(SLOT_COUNT);
  let cursor = 0;
  let completed = 0;

  async function worker() {
    while (cursor < ordered.length) {
      const index = cursor++;
      const entry = ordered[index];
      output[index] = await uploadImage(
        entry.file,
        entry.relativePath,
        index,
        subfolder,
      );
      completed += 1;
      setStatus(`画像をupload中: ${completed} / ${SLOT_COUNT}`);
    }
  }
  await Promise.all(Array.from({ length: 4 }, () => worker()));
  return output;
}

async function importZip(file, setStatus) {
  setStatus("ZIPを安全に展開しています…");
  const body = new FormData();
  body.append("archive", file, file.name);
  const response = await api.fetchApi("/wan-loop/batch/import-zip", {
    method: "POST",
    body,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload?.error ?? `ZIP import failed (${response.status})`);
  return {
    images: payload.images.map((item) => ({
      name: item.name,
      displayName: item.display_name,
    })),
    promptsText: payload.prompts_text,
  };
}

function parsePromptsText(rawText, filename = "prompts.txt") {
  const text = String(rawText).replace(/^\uFEFF/, "").trim();
  if (!text) throw new Error(`${filename} is empty`);

  let prompts;
  if (/^\[/.test(text)) {
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch (error) {
      throw new Error(`${filename} contains invalid JSON: ${error.message}`);
    }
    if (!Array.isArray(parsed)) throw new Error(`${filename} JSON must be an array`);
    prompts = parsed.map((item) => String(item).trim());
  } else if (/^\s*---+\s*$/m.test(text)) {
    prompts = text.split(/^\s*---+\s*$/m).map((item) => item.trim()).filter(Boolean);
  } else {
    prompts = text.split(/\r?\n/).map((item) => item.trim()).filter(Boolean);
    if (prompts.length === SLOT_COUNT && prompts.every((item) => /^\d{1,2}[.):]\s+/.test(item))) {
      prompts = prompts.map((item) => item.replace(/^\d{1,2}[.):]\s+/, ""));
    }
  }

  if (prompts.length !== SLOT_COUNT) {
    throw new Error(`${filename} must contain exactly 10 prompts; found ${prompts.length}`);
  }
  if (prompts.some((item) => !item)) throw new Error(`${filename} contains an empty prompt`);
  return prompts;
}

function assignImages(selector, images) {
  if (images.length !== SLOT_COUNT) throw new Error("image import did not return 10 items");
  const nodes = slotNodesOrThrow(selector);
  nodes.forEach((node, index) => {
    const widget = node.widgets?.find((item) => item.name === "image");
    if (!widget) throw new Error(`slot ${index + 1} image widget is missing`);
    if (Array.isArray(widget.options?.values) && !widget.options.values.includes(images[index].name)) {
      widget.options.values.push(images[index].name);
    }
    setWidgetValue(node, widget, images[index].name);
    node.title = `${String(index + 1).padStart(2, "0")}. ${images[index].displayName}`;
  });
  selector.wanLoopImportState.images = SLOT_COUNT;
  selector.graph?.setDirtyCanvas(true, true);
}

function assignPrompts(selector, prompts) {
  const nodes = slotNodesOrThrow(selector);
  nodes.forEach((node, index) => {
    const widget = node.widgets?.find((item) => item.name === "positive_prompt");
    if (!widget) throw new Error(`slot ${index + 1} prompt widget is missing`);
    setWidgetValue(node, widget, prompts[index]);
  });
  selector.wanLoopImportState.prompts = SLOT_COUNT;
  selector.graph?.setDirtyCanvas(true, true);
}

function readyText(selector) {
  const state = selector.wanLoopImportState;
  if (state.images === SLOT_COUNT && state.prompts === SLOT_COUNT) {
    return "準備完了: 画像10枚 + prompt 10件。QUEUE 10を押してください。";
  }
  return `画像 ${state.images}/${SLOT_COUNT} ・ prompt ${state.prompts}/${SLOT_COUNT}`;
}

function readAllDirectoryEntries(reader) {
  return new Promise((resolve, reject) => {
    const entries = [];
    const next = () => reader.readEntries((batch) => {
      if (!batch.length) return resolve(entries);
      entries.push(...batch);
      next();
    }, reject);
    next();
  });
}

async function walkEntry(entry, parent = "") {
  const relativePath = `${parent}${entry.name}`;
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    return [{ file, relativePath }];
  }
  if (!entry.isDirectory) return [];
  const children = await readAllDirectoryEntries(entry.createReader());
  const nested = await Promise.all(
    children.map((child) => walkEntry(child, `${relativePath}/`)),
  );
  return nested.flat();
}

async function collectDroppedFiles(dataTransfer) {
  const items = [...(dataTransfer?.items ?? [])];
  const entries = items
    .filter((item) => item.kind === "file")
    .map((item) => item.webkitGetAsEntry?.())
    .filter(Boolean);
  if (entries.length) {
    return (await Promise.all(entries.map((entry) => walkEntry(entry)))).flat();
  }
  return [...(dataTransfer?.files ?? [])].map((file) => ({
    file,
    relativePath: file.webkitRelativePath || file.name,
  }));
}

async function handleCollectedFiles(selector, entries, setStatus) {
  const usable = entries.filter((entry) => !/(^|\/)\.|(^|\/)__MACOSX\//.test(entry.relativePath));
  const archives = usable.filter((entry) => /\.zip$/i.test(entry.relativePath));
  const images = usable.filter((entry) => IMAGE_PATTERN.test(entry.relativePath));
  const promptFiles = usable.filter((entry) => PROMPT_PATTERN.test(entry.relativePath));
  if (archives.length > 1) throw new Error("drop only one ZIP at a time");
  if (archives.length && images.length) throw new Error("drop a folder/images or one ZIP, not both");
  if (promptFiles.length > 1) throw new Error("drop only one prompts.txt/JSON file");
  if (!archives.length && !images.length && !promptFiles.length) {
    throw new Error("no supported images, ZIP, or prompt text file was found");
  }

  let zipPrompts = null;
  if (archives.length) {
    const imported = await importZip(archives[0].file, setStatus);
    assignImages(selector, imported.images);
    zipPrompts = imported.promptsText;
  } else if (images.length) {
    setStatus(`画像10枚を確認しました。uploadを開始します…`);
    assignImages(selector, await uploadTenImages(images, setStatus));
  }

  if (promptFiles.length) {
    const item = promptFiles[0];
    assignPrompts(
      selector,
      parsePromptsText(await item.file.text(), item.relativePath),
    );
  } else if (zipPrompts) {
    assignPrompts(selector, parsePromptsText(zipPrompts, "ZIP/prompts.txt"));
  }
  setStatus(readyText(selector), "ok");
}

function makeBatchDropWidget(selector) {
  selector.wanLoopImportState = { images: 0, prompts: 0 };
  const root = document.createElement("div");
  root.style.cssText = [
    "box-sizing:border-box",
    "height:190px",
    "margin:4px",
    "padding:12px",
    "border:2px dashed #6f8cff",
    "border-radius:10px",
    "background:#151a28",
    "color:#e8ecff",
    "font:13px system-ui,sans-serif",
    "display:flex",
    "flex-direction:column",
    "gap:8px",
  ].join(";");
  root.innerHTML = `
    <strong style="font-size:14px">一括投入：画像フォルダ/ZIP + prompts.txt</strong>
    <div>ここへfolder、ZIP、またはprompt fileをdrop</div>
    <div data-status style="min-height:34px;color:#aebcff">画像 0/10 ・ prompt 0/10</div>
    <div data-buttons style="display:flex;gap:6px;flex-wrap:wrap"></div>
  `;
  const status = root.querySelector("[data-status]");
  const setStatus = (message, kind = "normal") => {
    status.textContent = message;
    status.style.color = kind === "error" ? "#ff8585" : kind === "ok" ? "#75e6a4" : "#aebcff";
  };
  const run = async (entries) => {
    try {
      root.style.pointerEvents = "none";
      await handleCollectedFiles(selector, entries, setStatus);
    } catch (error) {
      console.error("[WanLoopBatch] bulk import failed", error);
      setStatus(`ERROR: ${error?.message ?? error}`, "error");
      window.alert(`Batch import failed:\n\n${error?.message ?? error}`);
    } finally {
      root.style.pointerEvents = "auto";
    }
  };

  root.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    root.style.borderColor = "#75e6a4";
  });
  root.addEventListener("dragleave", () => { root.style.borderColor = "#6f8cff"; });
  root.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
    root.style.borderColor = "#6f8cff";
    void collectDroppedFiles(event.dataTransfer)
      .then(run)
      .catch((error) => {
        console.error("[WanLoopBatch] folder traversal failed", error);
        setStatus(`ERROR: ${error?.message ?? error}`, "error");
      });
  });

  const buttons = root.querySelector("[data-buttons]");
  function picker(label, options, collect) {
    const button = document.createElement("button");
    button.textContent = label;
    button.style.cssText = "padding:5px 9px;border:0;border-radius:6px;background:#34456f;color:white;cursor:pointer";
    const input = document.createElement("input");
    input.type = "file";
    Object.assign(input, options);
    input.style.display = "none";
    input.addEventListener("change", () => {
      const entries = [...input.files].map((file) => ({
        file,
        relativePath: file.webkitRelativePath || file.name,
      }));
      void run(collect ? collect(entries) : entries);
      input.value = "";
    });
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      input.click();
    });
    buttons.append(button, input);
  }
  picker("FOLDERを選択", { webkitdirectory: true, multiple: true });
  picker("ZIP/画像10枚", { accept: ".zip,image/*", multiple: true });
  picker("PROMPTS.TXT", { accept: ".txt,.json", multiple: false });
  return root;
}

async function queueTen(selector, button) {
  const problems = validateSlots(selector);
  if (problems.length) {
    window.alert(`Queue was not started:\n\n${problems.join("\n")}`);
    return;
  }

  const slotWidget = selector.widgets?.find((widget) => widget.name === "active_slot");
  const batchWidget = selector.widgets?.find((widget) => widget.name === "batch_id");
  const controlWidget = slotWidget?.linkedWidgets?.[0];
  if (!slotWidget || !batchWidget || !controlWidget) {
    window.alert("Queue controls were not initialized. Reload ComfyUI and reopen the workflow.");
    return;
  }

  button.disabled = true;
  button.name = "ADDING 10 JOBS TO QUEUE...";
  const batchId = newBatchId();
  try {
    setWidgetValue(selector, batchWidget, batchId);
    setWidgetValue(selector, slotWidget, 1);
    controlWidget.value = "increment";
    await app.queuePrompt(0, SLOT_COUNT);
    console.info(`[WanLoopBatch] queued ${SLOT_COUNT} sequential jobs for ${batchId}`);
  } catch (error) {
    console.error("[WanLoopBatch] queue failed", error);
    window.alert(`Failed to queue the 10-loop batch:\n\n${error?.message ?? error}`);
  } finally {
    setWidgetValue(selector, slotWidget, 1);
    controlWidget.value = "increment";
    button.disabled = false;
    button.name = "QUEUE 10 LOOPS (SEQUENTIAL)";
    selector.graph?.setDirtyCanvas(true, true);
  }
}

function triggerArchiveDownload(item) {
  const query = new URLSearchParams({
    filename: item.filename,
    subfolder: item.subfolder,
    type: item.type ?? "output",
  });
  const anchor = document.createElement("a");
  anchor.href = api.apiURL(`/view?${query.toString()}`);
  anchor.download = item.filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  console.info(`[WanLoopBatch] downloading ${item.filename}`);
}

app.registerExtension({
  name: "grawthings.WanLoopBatch10",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === "WanLoopQueueSelector") {
      const originalCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        const dropWidget = this.addDOMWidget(
          "batch_bulk_import",
          "div",
          makeBatchDropWidget(this),
          { serialize: false, hideOnZoom: false, getMinHeight: () => 190 },
        );
        dropWidget.serialize = false;
        const button = this.addWidget(
          "button",
          "QUEUE 10 LOOPS (SEQUENTIAL)",
          null,
          () => void queueTen(this, button),
          { serialize: false },
        );
        this.setSize([Math.max(this.size[0], 600), Math.max(this.size[1], 720)]);
        return result;
      };
    }

    if (nodeData.name === "WanLoopBatchFinalize") {
      const originalCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        const result = originalCreated?.apply(this, arguments);
        this.wanLoopDownloadButton = this.addWidget(
          "button",
          "DOWNLOAD LAST ZIP",
          null,
          () => {
            if (this.wanLoopLastArchive) triggerArchiveDownload(this.wanLoopLastArchive);
          },
          { serialize: false },
        );
        this.wanLoopDownloadButton.disabled = true;
        return result;
      };

      const originalExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        const result = originalExecuted?.apply(this, arguments);
        for (const item of message?.wan_loop_batch_download ?? []) {
          this.wanLoopLastArchive = item;
          if (this.wanLoopDownloadButton) {
            this.wanLoopDownloadButton.disabled = false;
            this.wanLoopDownloadButton.name = "DOWNLOAD LAST ZIP";
          }
          window.setTimeout(() => triggerArchiveDownload(item), 0);
        }
        return result;
      };
    }
  },
});
