"""Fail closed on the historical fake-quant entry points."""
def forbid_fake_quantization():
    from eval.compressors import head_ops, common
    from PhaseKV import quant

    def forbidden(*args, **kwargs):
        raise RuntimeError("Fake quantization is forbidden in this benchmark")

    head_ops.grouped_fake_quant = forbidden
    common.fake_quantize = forbidden
    for name in ("_grouped_affine_quant_dequant", "quantize_key", "quantize_value", "quantize_kv"):
        setattr(quant, name, forbidden)
