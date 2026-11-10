"""Model-defined EOS sets, including Llama assistant turn endings."""


def stop_token_ids(model, tokenizer):
    for source in (getattr(model, 'generation_config', None), getattr(model, 'config', None), tokenizer):
        eos = getattr(source, 'eos_token_id', None)
        if eos is not None:
            values = eos if isinstance(eos, (list, tuple, set)) else [eos]
            return {int(value) for value in values if value is not None}
    return set()
