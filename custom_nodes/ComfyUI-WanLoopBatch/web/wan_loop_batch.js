import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const SLOT_COUNT = 10;

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

function validateSlots(selector) {
  const nodes = linkedSlotNodes(selector);
  const problems = [];
  nodes.forEach((node, index) => {
    if (!node || node.type !== "WanLoopQueueSlot") {
      problems.push(`slot ${index + 1}: slot node is not connected`);
      return;
    }
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

async function queueTen(selector, button) {
  const problems = validateSlots(selector);
  if (problems.length) {
    window.alert(`Queue was not started:\n\n${problems.join("\n")}`);
    return;
  }

  const slotWidget = selector.widgets?.find(
    (widget) => widget.name === "active_slot",
  );
  const batchWidget = selector.widgets?.find(
    (widget) => widget.name === "batch_id",
  );
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
    // queuePrompt may return false when it accepted this request into its own
    // short frontend queue, so completion without an exception is success.
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
        const button = this.addWidget(
          "button",
          "QUEUE 10 LOOPS (SEQUENTIAL)",
          null,
          () => void queueTen(this, button),
          { serialize: false },
        );
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
            if (this.wanLoopLastArchive) {
              triggerArchiveDownload(this.wanLoopLastArchive);
            }
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
