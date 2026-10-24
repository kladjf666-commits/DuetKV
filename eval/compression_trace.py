"""Observe actual KV storage transitions, without reading GPU tensor values.

Token counts are layer-token positions (sum of sequence lengths over layers),
not unique text tokens, not multiplied by heads or K/V. K and V quantization
counts are separate. One event is one model-wide maintenance boundary; layer
counts identify how many layers changed. These are state changes, not GPU kernel
launch counts. Single-request initial prefill + single-token decode is supported.
"""
from collections import Counter

TRACE_VERSION = "kv_transitions_v1"


def _storage_bytes(tensors):
    # Count allocations once, including views/capacity padding.
    allocations = {}
    for t in tensors:
        s = t.untyped_storage()
        allocations[(str(t.device), s.data_ptr())] = s.nbytes()
    return sum(allocations.values())


def _pack_state(packs):
    return {
        "code_bytes": _storage_bytes([p.packed for p in packs]),
        "metadata_bytes": _storage_bytes([t for p in packs for t in (p.scale, p.mins)]),
        "code_dtypes": sorted({str(p.packed.dtype) for p in packs}),
        "bits": sorted({p.bits for p in packs}),
    }


def cache_state(cache):
    """Metadata-only snapshot; does not materialize/dequantize KV."""
    if hasattr(cache, "key_cache"):
        layers = []
        for k, v in zip(cache.key_cache, cache.value_cache):
            if k.shape != v.shape:
                raise AssertionError("Mismatched K/V shapes")
            layers.append({"tokens": k.shape[2], "kv_heads": k.shape[1],
                           "dtype": str(k.dtype), "packed_k_tokens": 0,
                           "packed_v_tokens": 0, "cold_tokens": 0, "cold_segments": 0})
        physical = _storage_bytes(cache.key_cache + cache.value_cache)
        return {"kind": "dense", "layers": layers, "physical_bytes": physical,
                "code_bytes": 0, "metadata_bytes": 0, "high_precision_capacity_bytes": physical,
                "code_dtypes": [], "bits": []}
    layers, packs, high = [], [], []
    for layer in cache.layers:
        if hasattr(layer, "key_residual"):
            k = layer.key.orig_shape[2] if layer.key is not None else 0
            v = layer.value.orig_shape[2] if layer.value is not None else 0
            layer_packs = [p for p in (layer.key, layer.value) if p is not None]
            high.extend((layer.key_residual, layer.value_residual))
            layers.append({"tokens": layer.length, "kv_heads": layer.key_residual.shape[1],
                           "packed_k_tokens": k, "packed_v_tokens": v,
                           "k_residual_tokens": layer.key_residual.shape[2],
                           "v_residual_tokens": layer.value_residual.shape[2],
                           "cold_tokens": 0, "cold_segments": 0})
            kind = "kivi_packed"
        else:
            segments = [layer.prompt, *layer.prompt_extensions, *layer.cold]
            layer_packs = [p for s in segments for p in (s.key, s.value)]
            high.extend((layer.hot_k, layer.hot_v))
            layers.append({"tokens": layer.physical_len, "kv_heads": layer.hot_k.shape[1],
                           "packed_k_tokens": layer.prompt_len + layer.cold_len,
                           "packed_v_tokens": layer.prompt_len + layer.cold_len,
                           "prompt_tokens": layer.prompt_len, "cold_tokens": layer.cold_len,
                           "cold_segments": len(layer.cold), "hot_tokens": layer.hot_len})
            kind = "phasekv_packed"
        packs.extend(layer_packs)
    packed = _pack_state(packs)
    high_bytes = _storage_bytes(high)
    return {"kind": kind, "layers": layers, **packed,
            "high_precision_capacity_bytes": high_bytes,
            "physical_bytes": packed["code_bytes"] + packed["metadata_bytes"] + high_bytes}


class CompressionTrace:
    def __init__(self, method, implementation=None):
        self.method = method
        self.implementation = implementation or {}
        self.events = []
        self.checks = Counter()
        self.initial = None
        self.final = None

    def observe(self, stage, step, before, after, *, appended_tokens=0):
        """step=0 for prefill; decode steps are 1-based cache appends."""
        if stage not in ("prefill", "decode"):
            raise ValueError(stage)
        if len(before["layers"]) != len(after["layers"]):
            raise AssertionError("Layer count changed")
        self.checks[stage] += 1
        if self.initial is None:
            self.initial = before
        changes = []
        for i, (old, new) in enumerate(zip(before["layers"], after["layers"])):
            evicted = old["tokens"] + appended_tokens - new["tokens"]
            qk = new["packed_k_tokens"] - old["packed_k_tokens"]
            qv = new["packed_v_tokens"] - old["packed_v_tokens"]
            cooled = new["cold_tokens"] - old["cold_tokens"]
            batches = new["cold_segments"] - old["cold_segments"]
            if min(evicted, qk, qv, cooled, batches) < 0:
                raise AssertionError("Unsupported cache transition or incorrect append accounting")
            if evicted or qk or qv or batches:
                changes.append({"layer": i, "tokens_before": old["tokens"],
                                "tokens_after": new["tokens"], "evicted_tokens": evicted,
                                "quantized_k_tokens": qk, "quantized_v_tokens": qv,
                                "cooled_tokens": cooled, "cold_batches": batches,
                                "hot_tokens_processed": cooled + evicted if batches else 0})
        if changes:
            actions = []
            for field, name in (("evicted_tokens", "eviction"), ("quantized_k_tokens", "key_quantization"),
                                ("quantized_v_tokens", "value_quantization"), ("cold_batches", "hot_to_cold")):
                if any(c[field] for c in changes):
                    actions.append(name)
            self.events.append({"stage": stage, "decode_step": step, "actions": actions,
                                "physical_bytes_before": before["physical_bytes"],
                                "physical_bytes_after": after["physical_bytes"], "layers": changes})
        self.final = after

    def export(self):
        stages = {}
        for stage in ("prefill", "decode"):
            events = [e for e in self.events if e["stage"] == stage]
            sums = {f: sum(c[f] for e in events for c in e["layers"])
                    for f in ("evicted_tokens", "quantized_k_tokens", "quantized_v_tokens",
                              "cooled_tokens", "cold_batches", "hot_tokens_processed")}
            stages[stage] = {"checks": self.checks[stage], "triggered_steps": len(events),
                             "first_trigger_step": events[0]["decode_step"] if events else None,
                             "events_by_action": dict(Counter(a for e in events for a in e["actions"])),
                             "layer_token_totals": sums}
            if events:
                reason = "observed_storage_change"
            elif not self.checks[stage]:
                reason = "stage_not_executed"
            elif self.method == "full":
                reason = "full_cache_baseline"
            elif stage == "decode" and self.method in ("snapkv", "pyramidkv"):
                reason = "prefill_only_method"
            elif self.method == "scope":
                reason = ("allkv_prompt_preserved" if stage == "prefill"
                          else "generated_cache_not_above_decode_budget")
            elif stage == "decode" and self.method == "phasekv":
                reason = "hot_buffer_below_cooling_threshold"
            elif self.method == "kivi":
                reason = "no_packable_prefix_or_residual_flush"
            else:
                reason = "no_observed_storage_change"
            stages[stage]["reason"] = reason
        if not self.checks["prefill"]:
            status = "not_observed"
        elif self.events:
            status = "triggered"
        elif self.method == "full":
            status = "baseline_no_compression"
        else:
            status = "not_triggered"
        return {"version": TRACE_VERSION, "method": self.method, "status": status,
                "measurement": "observed_storage_transitions_not_kernel_launch_counts",
                "token_count_unit": "sum_of_sequence_positions_over_layers_not_multiplied_by_heads",
                "step_convention": "prefill=0; decode=1-based_cache_append; final_sampled_token_not_written",
                "implementation": self.implementation, "summary": stages,
                "initial_state": self.initial, "final_state": self.final, "events": self.events}
