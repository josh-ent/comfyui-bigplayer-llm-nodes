import { app } from "../../scripts/app.js";

const TARGET_NODES = new Set(["BigPlayerPromptSimple", "BigPlayerPromptSplit"]);

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name) ?? null;
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

app.registerExtension({
  name: "BigPlayer.DynamicProviderModels",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODES.has(nodeData.name)) {
      return;
    }

    const providerModels = nodeData.input?.required?.provider_model?.[1]?.provider_models;
    if (!providerModels) {
      return;
    }

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
  },
});
