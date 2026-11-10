"""Call vendored official scorers; additional diagnostics are named separately."""
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
_MODULES = {}


def load(name, relative):
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        _MODULES[name] = obj
    return _MODULES[name]


def score(sample, prediction):
    suite, task = sample['suite'], sample['task']
    if suite == 'performance':
        return {}
    answers = sample['answers']
    if suite == 'longbench':
        mod = load('official_longbench_metrics', 'vendor/SnapKV/experiments/LongBench/metrics.py')
        name = ('rouge_score' if task in ('gov_report','qmsum','multi_news') else
                'code_sim_score' if task in ('lcc','repobench-p') else 'qa_f1_score')
        value = max(getattr(mod, name)(prediction, answer) for answer in answers)
        return {'score': 100*value, 'metric': name}
    if suite == 'ruler':
        mod = load('official_ruler_metrics', 'bench_sources/RULER/scripts/eval/synthetic/constants.py')
        prediction = re.sub(r'[\x00-\x1f]', '\n', prediction.strip()).strip()
        value = mod.TASKS[sample['ruler_base_task']]['metric_fn']([prediction], [answers])
        return {'score': value, 'metric': 'official_string_match_all'}
    if suite != 'longgenbench':
        raise ValueError(suite)
    mod = load('official_longgenbench_metrics', 'vendor/SCOPE/eval_longgenbench.py')
    value = (mod.compare_answers if task == 'gsm8k' else mod.compare_choices)(prediction, answers)
    # Official scoring aligns a list of regex matches, which can shift after a
    # missing answer. Retain it, but separately report a number-aware diagnostic.
    blocks = re.findall(r'Answer_(\d+)\s*:(.*?)(?=Answer_\d+\s*:|\Z)', prediction, re.S)
    by_number = {}
    for number, block in blocks:
        parsed = re.search(r'answer is (\d+)' if task == 'gsm8k' else r'answer is \((.*?)\)', block)
        if parsed and int(number) not in by_number:
            by_number[int(number)] = parsed[1]
    start = sample['first_answer_number']
    expected = [mod.extract_final_answer(a) for a in answers] if task == 'gsm8k' else answers
    correct = [by_number.get(start+i) == a and a is not None for i,a in enumerate(expected)]
    completed = sum(start+i in by_number for i in range(len(answers)))
    tail = correct[len(correct)*2//3:]
    return {'score': 100*value, 'metric': 'official_compound_accuracy',
            'indexed_accuracy_diagnostic': 100*sum(correct)/len(correct),
            'answered_questions': completed, 'total_questions': len(answers),
            'unanswered_questions': len(answers)-completed,
            'completion_pct': 100*completed/len(answers),
            'last_third_indexed_accuracy': 100*sum(tail)/len(tail)}
