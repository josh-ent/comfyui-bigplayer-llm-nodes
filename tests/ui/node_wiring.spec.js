const { expect, test } = require("@playwright/test");

const expectedNodes = [
  ["BigPlayerLLMProvider", "BigPlayer LLM Provider"],
  ["BigPlayerNaturalLanguageRoot", "BigPlayer Natural Language Root"],
  ["BigPlayerBasicPrompt", "BigPlayer Basic Prompt"],
  ["BigPlayerSplitPrompt", "BigPlayer Split Prompt"],
  ["BigPlayerKSamplerConfig", "BigPlayer KSampler Config"],
  ["BigPlayerCheckpointPicker", "BigPlayer Checkpoint Picker"],
  ["BigPlayerCheckpointState", "BigPlayer Checkpoint State"],
  ["BigPlayerLoRAState", "BigPlayer LoRA State"],
  ["BigPlayerControlNetState", "BigPlayer ControlNet State"],
];

const typeColors = {
  BIGPLAYER_LLM_PROVIDER: "#b76a5f",
  BIGPLAYER_LLM_SESSION: "#5f8f8b",
  BIGPLAYER_PRESET_CONFIG: "#8b6fb0",
};

async function waitForEditor(page) {
  await page.goto("/", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(
    () => Boolean(globalThis.app?.graph) && Boolean(globalThis.LiteGraph?.registered_node_types?.BigPlayerLLMProvider),
    null,
    { timeout: 20_000 },
  );
}

test.describe("BigPlayer ComfyUI frontend wiring", () => {
  test.beforeEach(async ({ page }) => {
    await waitForEditor(page);
  });

  test("registers the BigPlayer nodes in LiteGraph", async ({ page }) => {
    const registered = await page.evaluate((nodeTypes) => {
      globalThis.app.graph.clear();
      return nodeTypes.map((type) => {
        const registeredType = globalThis.LiteGraph.registered_node_types[type];
        const node = registeredType ? globalThis.LiteGraph.createNode(type) : null;
        return node
          ? {
              type,
              title: node.title,
            }
          : null;
      });
    }, expectedNodes.map(([type]) => type));

    expect(registered).toHaveLength(expectedNodes.length);
    for (const [index, [type, title]] of expectedNodes.entries()) {
      expect(registered[index]).toEqual({ type, title });
    }
  });

  test("updates provider-model options when the provider changes", async ({ page }) => {
    const state = await page.evaluate(async () => {
      const objectInfo = await fetch("/object_info").then((response) => response.json());
      const providerModels =
        objectInfo.BigPlayerLLMProvider.input.required.provider_model[1].provider_models;

      globalThis.app.graph.clear();
      const node = globalThis.LiteGraph.createNode("BigPlayerLLMProvider");
      globalThis.app.graph.add(node);

      const getWidget = (name) => node.widgets.find((widget) => widget.name === name);
      const provider = getWidget("provider");
      const providerModel = getWidget("provider_model");
      if (!provider?.callback || !providerModel?.options?.values) {
        throw new Error("Provider widgets were not initialized.");
      }

      const snapshot = () => ({
        provider: provider.value,
        selectedModel: providerModel.value,
        availableModels: [...providerModel.options.values],
      });

      const initial = snapshot();
      provider.value = "No Provider";
      provider.callback();
      const noProvider = snapshot();
      provider.value = "xAI";
      provider.callback();
      const xai = snapshot();

      return { providerModels, initial, noProvider, xai };
    });

    expect(state.initial.availableModels).toEqual(state.providerModels[state.initial.provider]);
    expect(state.noProvider.availableModels).toEqual(state.providerModels["No Provider"]);
    expect(state.noProvider.selectedModel).toBe(state.providerModels["No Provider"][0]);
    expect(state.xai.availableModels).toEqual(state.providerModels.xAI);
    expect(state.xai.selectedModel).toBe(state.providerModels.xAI[0]);
  });

  test("reapplies provider-model options when a provider node is reconfigured", async ({ page }) => {
    const reloaded = await page.evaluate(async () => {
      const objectInfo = await fetch("/object_info").then((response) => response.json());
      const providerModels =
        objectInfo.BigPlayerLLMProvider.input.required.provider_model[1].provider_models;

      globalThis.app.graph.clear();
      const node = globalThis.LiteGraph.createNode("BigPlayerLLMProvider");
      globalThis.app.graph.add(node);

      const getWidget = (owner, name) => owner.widgets.find((widget) => widget.name === name);
      const provider = getWidget(node, "provider");
      const providerModel = getWidget(node, "provider_model");
      if (!provider?.callback) {
        throw new Error("Provider widget callback was not installed.");
      }

      provider.value = "No Provider";
      provider.callback();
      providerModel.value = providerModels["No Provider"][1];

      const serialized = node.serialize();
      const restored = globalThis.LiteGraph.createNode("BigPlayerLLMProvider");
      globalThis.app.graph.add(restored);
      restored.configure(serialized);

      const restoredProvider = getWidget(restored, "provider");
      const restoredModel = getWidget(restored, "provider_model");
      return {
        provider: restoredProvider.value,
        selectedModel: restoredModel.value,
        availableModels: [...restoredModel.options.values],
      };
    });

    expect(reloaded.provider).toBe("No Provider");
    expect(reloaded.availableModels).toEqual(["Positive", "Negative"]);
    expect(reloaded.selectedModel).toBe("Negative");
  });

  test("publishes the intended BigPlayer category structure", async ({ page }) => {
    const categories = await page.evaluate(async () => {
      const objectInfo = await fetch("/object_info").then((response) => response.json());
      return {
        provider: objectInfo.BigPlayerLLMProvider.category,
        root: objectInfo.BigPlayerNaturalLanguageRoot.category,
        basicPrompt: objectInfo.BigPlayerBasicPrompt.category,
        splitPrompt: objectInfo.BigPlayerSplitPrompt.category,
        ksamplerConfig: objectInfo.BigPlayerKSamplerConfig.category,
        checkpointPicker: objectInfo.BigPlayerCheckpointPicker.category,
        checkpointState: objectInfo.BigPlayerCheckpointState.category,
      };
    });

    expect(categories).toEqual({
      provider: "BigPlayer",
      root: "BigPlayer/Prompting",
      basicPrompt: "BigPlayer/Prompting/Capabilities",
      splitPrompt: "BigPlayer/Prompting/Capabilities",
      ksamplerConfig: "BigPlayer/Prompting/Capabilities",
      checkpointPicker: "BigPlayer/Prompting/Capabilities",
      checkpointState: "BigPlayer/State Indication",
    });
  });

  test("publishes renamed linked-input sockets with clear tooltips", async ({ page }) => {
    const state = await page.evaluate(async () => {
      const objectInfo = await fetch("/object_info").then((response) => response.json());
      const loraOptional = objectInfo.BigPlayerLoRAState.input.optional;
      const controlnetOptional = objectInfo.BigPlayerControlNetState.input.optional;
      return {
        loraAlso: loraOptional.lora_syntax_also,
        controlnetsAlso: controlnetOptional.controlnets_also,
        hasLegacyLora: Boolean(loraOptional.lora_syntax_input),
        hasLegacyControlnet: Boolean(controlnetOptional.controlnets_input),
      };
    });

    expect(state.hasLegacyLora).toBe(false);
    expect(state.hasLegacyControlnet).toBe(false);
    expect(state.loraAlso[1].tooltip).toContain("concatenates it onto `lora_syntax`");
    expect(state.loraAlso[1].tooltip).toContain("linked entries win ties where needed");
    expect(state.controlnetsAlso[1].tooltip).toContain("concatenates it onto `controlnets`");
    expect(state.controlnetsAlso[1].tooltip).toContain("linked entries win ties where needed");
  });

  test("applies BigPlayer concept colors to sockets and links", async ({ page }) => {
    const state = await page.evaluate((expectedTypeColors) => {
      globalThis.app.graph.clear();
      const providerNode = globalThis.LiteGraph.createNode("BigPlayerLLMProvider");
      const rootNode = globalThis.LiteGraph.createNode("BigPlayerNaturalLanguageRoot");
      const checkpointStateNode = globalThis.LiteGraph.createNode("BigPlayerCheckpointState");
      globalThis.app.graph.add(providerNode);
      globalThis.app.graph.add(rootNode);
      globalThis.app.graph.add(checkpointStateNode);
      const findSlot = (slots, name) => slots.find((slot) => slot.name === name);

      return {
        typeColors: Object.fromEntries(
          Object.keys(expectedTypeColors).map((key) => [key, globalThis.LGraphCanvas.link_type_colors[key]]),
        ),
        canvasTypeColors: {
          byType: Object.fromEntries(
            Object.keys(expectedTypeColors).map((key) => [key, globalThis.app.canvas.default_connection_color_byType[key]]),
          ),
          byTypeOn: Object.fromEntries(
            Object.keys(expectedTypeColors).map((key) => [key, globalThis.app.canvas.default_connection_color_byTypeOn[key]]),
          ),
          byTypeOff: Object.fromEntries(
            Object.keys(expectedTypeColors).map((key) => [key, globalThis.app.canvas.default_connection_color_byTypeOff[key]]),
          ),
        },
        colourGetterColors: {
          providerConfig: {
            connected: globalThis.app.canvas.colourGetter.getConnectedColor("BIGPLAYER_LLM_PROVIDER"),
            disconnected: globalThis.app.canvas.colourGetter.getDisconnectedColor("BIGPLAYER_LLM_PROVIDER"),
          },
          session: {
            connected: globalThis.app.canvas.colourGetter.getConnectedColor("BIGPLAYER_LLM_SESSION"),
            disconnected: globalThis.app.canvas.colourGetter.getDisconnectedColor("BIGPLAYER_LLM_SESSION"),
          },
          presetConfig: {
            connected: globalThis.app.canvas.colourGetter.getConnectedColor("BIGPLAYER_PRESET_CONFIG"),
            disconnected: globalThis.app.canvas.colourGetter.getDisconnectedColor("BIGPLAYER_PRESET_CONFIG"),
          },
        },
      };
    }, typeColors);

    expect(state.typeColors).toEqual(typeColors);
    expect(state.canvasTypeColors.byType).toEqual(typeColors);
    expect(state.canvasTypeColors.byTypeOn).toEqual(typeColors);
    expect(state.canvasTypeColors.byTypeOff).toEqual(typeColors);
    expect(state.colourGetterColors.providerConfig).toEqual({
      connected: typeColors.BIGPLAYER_LLM_PROVIDER,
      disconnected: typeColors.BIGPLAYER_LLM_PROVIDER,
    });
    expect(state.colourGetterColors.session).toEqual({
      connected: typeColors.BIGPLAYER_LLM_SESSION,
      disconnected: typeColors.BIGPLAYER_LLM_SESSION,
    });
    expect(state.colourGetterColors.presetConfig).toEqual({
      connected: typeColors.BIGPLAYER_PRESET_CONFIG,
      disconnected: typeColors.BIGPLAYER_PRESET_CONFIG,
    });
  });
});
