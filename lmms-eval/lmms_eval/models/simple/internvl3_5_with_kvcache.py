from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.internvl3_with_kvcache import InternVL3_with_kvcache


@register_model("internvl3_5_with_kvcache")
class InternVL3_5_with_kvcache(InternVL3_with_kvcache):
    """InternVL3.5 model wrapper with KV-cache compression support.

    Uses the same implementation as InternVL3_with_kvcache since both share
    identical interfaces. Default pretrained model is set to InternVL3_5-8B.
    """

    def __init__(
        self,
        pretrained: str = "OpenGVLab/InternVL3_5-8B",
        **kwargs,
    ):
        super().__init__(pretrained=pretrained, **kwargs)
