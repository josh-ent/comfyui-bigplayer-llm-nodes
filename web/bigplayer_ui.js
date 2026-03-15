import { app } from "../../scripts/app.js";

const PROVIDER_MODEL_NODES = new Set(["BigPlayerLLMProvider"]);

const TYPE_COLORS = {
  BIGPLAYER_LLM_PROVIDER: "#b76a5f",
  BIGPLAYER_LLM_SESSION: "#5f8f8b",
  BIGPLAYER_PRESET_CONFIG: "#8b6fb0",
};

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function assignTypeColors(target) {
  if (!target || typeof target !== "object") {
    return;
  }
  Object.assign(target, TYPE_COLORS);
}

function ensureTypeColorMap(target, key) {
  if (!target || typeof target !== "object") {
    return null;
  }
  if (!target[key] || typeof target[key] !== "object") {
    target[key] = {};
  }
  return target[key];
}

function registerTypeColors() {
  assignTypeColors(globalThis.LGraphCanvas?.link_type_colors);
  assignTypeColors(ensureTypeColorMap(globalThis.LGraphCanvas, "DEFAULT_CONNECTION_COLORS_BY_TYPE"));
  assignTypeColors(ensureTypeColorMap(globalThis.LGraphCanvas, "DEFAULT_CONNECTION_COLORS_BY_TYPE_OFF"));
  assignTypeColors(ensureTypeColorMap(globalThis.LGraphCanvas, "DEFAULT_CONNECTION_COLORS_BY_TYPE_ON"));
  assignTypeColors(ensureTypeColorMap(app.canvas, "default_connection_color_byType"));
  assignTypeColors(ensureTypeColorMap(app.canvas, "default_connection_color_byTypeOff"));
  assignTypeColors(ensureTypeColorMap(app.canvas, "default_connection_color_byTypeOn"));
}

function applySlotTypeColors(node) {
  for (const slots of [node.inputs, node.outputs]) {
    if (!Array.isArray(slots)) {
      continue;
    }
    for (const slot of slots) {
      const color = TYPE_COLORS[slot?.type];
      if (!color) {
        continue;
      }
      slot.color = color;
      slot.color_on = color;
      slot.color_off = color;
      slot.link_color = color;
    }
  }
  node.setDirtyCanvas(true, true);
}

function applyProviderModelOptions(node, providerModels) {
  const providerWidget = getWidget(node, "provider");
  const providerModelWidget = getWidget(node, "provider_model");
  if (!providerWidget || !providerModelWidget) {
    return;
  }

  const selectedProvider = providerWidget.value;
  const models = providerModels[selectedProvider] ?? [];
  providerModelWidget.options = providerModelWidget.options || {};
  providerModelWidget.options.values = models;

  if (!models.includes(providerModelWidget.value)) {
    providerModelWidget.value = models[0] ?? "";
  }

  node.setDirtyCanvas(true, true);
}

function installProviderWidgetBehavior(nodeType, providerModels) {
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
    const providerWidget = getWidget(this, "provider");
    if (providerWidget) {
      const originalCallback = providerWidget.callback;
      providerWidget.callback = (...args) => {
        if (originalCallback) {
          originalCallback.apply(providerWidget, args);
        }
        applyProviderModelOptions(this, providerModels);
      };
    }
    applyProviderModelOptions(this, providerModels);
    return result;
  };

  const originalOnConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
    applyProviderModelOptions(this, providerModels);
    return result;
  };
}

function installTypeColorBehavior(nodeType) {
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
    registerTypeColors();
    applySlotTypeColors(this);
    return result;
  };

  const originalOnConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
    registerTypeColors();
    applySlotTypeColors(this);
    return result;
  };
}

app.registerExtension({
  name: "BigPlayer.UI",
  async setup() {
    registerTypeColors();
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const nodeName = nodeData.name;

    if (PROVIDER_MODEL_NODES.has(nodeName)) {
      const providerModels = nodeData.input?.required?.provider_model?.[1]?.provider_models;
      if (providerModels) {
        installProviderWidgetBehavior(nodeType, providerModels);
      }
    }

    installTypeColorBehavior(nodeType);
  },
});
