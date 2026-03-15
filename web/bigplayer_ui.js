import { app } from "../../scripts/app.js";

const PROVIDER_MODEL_NODES = new Set(["BigPlayerLLMProvider"]);

const TYPE_COLORS = {
  BIGPLAYER_LLM_PROVIDER: "#2f8bbd",
  BIGPLAYER_LLM_SESSION: "#c68728",
  BIGPLAYER_PRESET_CONFIG: "#669a4a",
};

const NODE_STYLES = {
  BigPlayerLLMProvider: { color: "#2f8bbd", bgcolor: "#1b3f52" },
  BigPlayerNaturalLanguageRoot: { color: "#c68728", bgcolor: "#574018" },
  BigPlayerBasicPrompt: { color: "#c68728", bgcolor: "#574018" },
  BigPlayerSplitPrompt: { color: "#c68728", bgcolor: "#574018" },
  BigPlayerKSamplerConfig: { color: "#c68728", bgcolor: "#574018" },
  BigPlayerCheckpointPicker: { color: "#c68728", bgcolor: "#574018" },
  BigPlayerCheckpointState: { color: "#669a4a", bgcolor: "#2a4020" },
  BigPlayerLoRAState: { color: "#669a4a", bgcolor: "#2a4020" },
  BigPlayerControlNetState: { color: "#669a4a", bgcolor: "#2a4020" },
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

function registerTypeColors() {
  assignTypeColors(globalThis.LGraphCanvas?.link_type_colors);
  assignTypeColors(globalThis.LGraphCanvas?.DEFAULT_CONNECTION_COLORS_BY_TYPE);
  assignTypeColors(globalThis.LGraphCanvas?.DEFAULT_CONNECTION_COLORS_BY_TYPE_OFF);
  assignTypeColors(globalThis.LGraphCanvas?.DEFAULT_CONNECTION_COLORS_BY_TYPE_ON);
  assignTypeColors(app.canvas?.default_connection_color_byType);
  assignTypeColors(app.canvas?.default_connection_color_byTypeOff);
  assignTypeColors(app.canvas?.default_connection_color_byTypeOn);
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

function applyNodeStyle(node, nodeName) {
  const style = NODE_STYLES[nodeName];
  if (!style) {
    return;
  }
  node.color = style.color;
  node.bgcolor = style.bgcolor;
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

function installNodeStyleBehavior(nodeType, nodeName) {
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
    registerTypeColors();
    applyNodeStyle(this, nodeName);
    return result;
  };

  const originalOnConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function () {
    const result = originalOnConfigure ? originalOnConfigure.apply(this, arguments) : undefined;
    registerTypeColors();
    applyNodeStyle(this, nodeName);
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

    if (NODE_STYLES[nodeName]) {
      installNodeStyleBehavior(nodeType, nodeName);
    }
  },
});
