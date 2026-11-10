"""CPU diagnostics, called only after all CUDA timing/peak reads are finished."""
from collections import Counter


def describe_outputs(outputs, eos_ids):
    result = []
    for tokens in outputs:
        first = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        visible = tokens if first is None else tokens[:first]
        grams = Counter(tuple(visible[i:i + 4]) for i in range(max(0, len(visible) - 3)))
        total = sum(grams.values())
        result.append(dict(first_eos_token_1based=None if first is None else first + 1,
            forced_tokens_after_first_eos=0 if first is None else len(tokens) - first - 1,
            tokens_before_first_eos=len(visible),
            repeated_4gram_fraction_before_eos=None if not total else (total - len(grams)) / total,
            task_accuracy=None, note='EOS/repetition diagnostics only; not a task quality score.'))
    return result
