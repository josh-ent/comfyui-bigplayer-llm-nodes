class _FakeModel:
    def __init__(self, model_name: str):
        self.cached_patcher_init = (object(), (f"/models/{model_name}",))


class BigPlayerTestModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "sdxl-base-1.0.safetensors"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "build"
    CATEGORY = "Testing/BigPlayer"

    def build(self, model_name):
        return (_FakeModel(model_name),)


class BigPlayerTestSink:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_1": ("STRING", {"forceInput": True}),
                "value_2": ("STRING", {"forceInput": True}),
                "value_3": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("value_1", "value_2", "value_3")
    FUNCTION = "collect"
    CATEGORY = "Testing/BigPlayer"
    OUTPUT_NODE = True

    def collect(self, value_1, value_2, value_3):
        return {
            "ui": {
                "value_1": [value_1],
                "value_2": [value_2],
                "value_3": [value_3],
            },
            "result": (value_1, value_2, value_3),
        }


class BigPlayerTestSplitSink:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value_1": ("STRING", {"forceInput": True}),
                "value_2": ("STRING", {"forceInput": True}),
                "value_3": ("STRING", {"forceInput": True}),
                "value_4": ("STRING", {"forceInput": True}),
                "value_5": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("value_1", "value_2", "value_3", "value_4", "value_5")
    FUNCTION = "collect"
    CATEGORY = "Testing/BigPlayer"
    OUTPUT_NODE = True

    def collect(self, value_1, value_2, value_3, value_4, value_5):
        return {
            "ui": {
                "value_1": [value_1],
                "value_2": [value_2],
                "value_3": [value_3],
                "value_4": [value_4],
                "value_5": [value_5],
            },
            "result": (value_1, value_2, value_3, value_4, value_5),
        }


NODE_CLASS_MAPPINGS = {
    "BigPlayerTestModel": BigPlayerTestModel,
    "BigPlayerTestSink": BigPlayerTestSink,
    "BigPlayerTestSplitSink": BigPlayerTestSplitSink,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BigPlayerTestModel": "BigPlayer Test Model",
    "BigPlayerTestSink": "BigPlayer Test Sink",
    "BigPlayerTestSplitSink": "BigPlayer Test Split Sink",
}
