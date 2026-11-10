"""Fixed-prefix Llama replay of early LongGenBench failures (one real layer resident at a time)."""
import collections
import gzip
import hashlib
import json
import re
import statistics
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import phasekv_optimization_support as support
support.bootstrap()
from models.Llama.llama_attention import PhaseKVLlamaAttention
from PhaseKV import hetero_cache
from PhaseKV.kernels.packing import quantize_pack_fast, unpack_dequantize
from PhaseKV.optimizations import stage_configs
from PhaseKV.phasekv_cache import PhaseKVCache
from PhaseKV.prompt_tier import select_prompt_indices

MODEL = ROOT / 'weights/Llama-3.1-8B-Instruct/snapshots/master'
FROZEN = ROOT / 'official_llama/bench_assets/frozen/samples.jsonl.gz'
OUT = ROOT / 'results/longgen_quality_probe'
ARMS = ('all_fp16', 'all_4bit', 'keep70_fp16', 'keep70_4bit', 'keep70_packed')
PIN = ('keep70_4bit_pin_question', 'keep70_4bit_pin_control')


def divergence(x, y):
    return next((i for i, (a, b) in enumerate(zip(x, y), 1) if a != b),
                None if len(x) == len(y) else min(len(x), len(y)) + 1)


def read_records(lines, method):
    rows = {}
    for line in lines:
        row = json.loads(line)
        if row['suite'] == 'longgenbench' and row['method'] == method:
            assert row['status'] == 'ok' and row['repeat'] == 0 and row['id'] not in rows
            rows[row['id']] = row
    assert len(rows) == 48
    return rows


def indexed_choices(text):
    answers = {}
    for number, block in re.findall(r'Answer_(\d+)\s*:(.*?)(?=Answer_\d+\s*:|\Z)', text, re.S):
        match = re.search(r'answer is \((.*?)\)', block)
        if match:
            answers.setdefault(int(number), match[1])
    return answers


def archive_audit():
    archive = ROOT / 'PhaseKV与Full消融/PhaseKV_4090_keep80_results.zip'
    with zipfile.ZipFile(archive) as z, z.open('rtx4090_channel_keep80_full_v1/records.jsonl') as f:
        phase = read_records(f, 'phasekv_channel_k_d080')
    native = ROOT / '不同算法的不同数据集测试/native_llama_20261029/records.jsonl'
    with native.open('rb') as f:
        full = read_records(f, 'full')
    with native.open('rb') as f:
        old = read_records(f, 'phasekv')
    with gzip.open(FROZEN, 'rt', encoding='utf-8') as f:
        samples = {s['id']: s for line in f if (s := json.loads(line))['id'] in full}
    audit = {}
    matched = collections.defaultdict(collections.Counter)
    for key, s in samples.items():
        a, b, c = full[key], phase[key], old[key]
        assert hashlib.sha256(json.dumps(s['input_ids']).encode()).hexdigest() == s['input_sha256']
        assert a['input_sha256'] == b['input_sha256'] == c['input_sha256'] == s['input_sha256']
        detail = {}
        for name, row in [('full', a), ('keep80', b), ('old100', c)]:
            repeated = collections.Counter(re.findall(r'\bAnswer_(\d+):', row['prediction'])).most_common(1)
            header, count = repeated[0] if repeated else (None, 0)
            detail[name] = dict(first_diff_vs_full=divergence(a['output_ids'], row['output_ids']),
                                top_header=header, header_count=count, score=row['score'],
                                output_tokens=row['output_tokens'], answered=row['answered_questions'],
                                severe=count >= 100 and not row['answered_questions'] and row['hit_output_limit'])
        if s['task'] in ('mmlu', 'csqa'):
            fa, pa = indexed_choices(a['prediction']), indexed_choices(b['prediction'])
            for i, correct in enumerate(s['answers'], s['first_answer_number']):
                if i in fa and i in pa:
                    matched[s['task']].update(n=1, full=int(fa[i] == correct), keep80=int(pa[i] == correct))
        audit[key] = detail
    return samples, phase, full, audit, dict(matched)


def question_regions(tokenizer, sample):
    ids = sample['input_ids']
    text = tokenizer.decode(ids, skip_special_tokens=False)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert encoded.ids == ids
    starts = {int(m[1]): m.start() for m in re.finditer(r'(?m)^Question_(\d+):', text)}
    number = sample['first_answer_number']
    def position(char):
        return next(i for i, (_, end) in enumerate(encoded.offsets) if end > char)
    a, b = position(starts[number]), position(starts[number + 1])
    control = position(starts[number + 10])
    assert b < control and control + (b-a) <= len(ids)-32
    return (a, b), (control, control + (b-a))


def prepare_cases(embed, rotary, tokenizer, samples, phase, full, audit, *, all_cases=False):
    cases = []
    for key in sorted(samples):
        sample, a, b = samples[key], full[key], phase[key]
        severe = audit[key]['keep80']['severe']
        escaped = key == 'longgenbench/gsm8k/15'
        mmlu = sample['task'] == 'mmlu'
        if not (all_cases or severe or escaped or mmlu or key in ('longgenbench/gsm8k/24','longgenbench/csqa/12')):
            continue
        div = audit[key]['keep80']['first_diff_vs_full']
        if severe or escaped:
            assert div == 5 and a['output_ids'][:4] == b['output_ids'][:4]
            teacher = b['output_ids'][:4] * 3
            positions = [4, 8, 12]  # number of forced tokens before prediction
        else:
            teacher = a['output_ids'][:div-1]
            assert teacher == b['output_ids'][:div-1]
            positions = [len(teacher)]
        length = len(sample['input_ids'])
        x = embed[torch.tensor([sample['input_ids']], device='cuda')]
        y = embed[torch.tensor([teacher], device='cuda')]
        pos = torch.arange(length+len(teacher), device='cuda')
        mask = torch.zeros((1, 1, 32, length), device='cuda', dtype=torch.float16)
        mask.masked_fill_(pos[:length][None] > pos[length-32:length,None], torch.finfo(mask.dtype).min)
        rope = [rotary(x, z[None]) for z in (pos[:length-32], pos[length-32:length], pos[length:])]
        arms = ARMS + PIN if severe or escaped else ARMS
        cases.append(dict(id=key, cohort='severe' if severe else 'escaped' if escaped else 'mmlu' if mmlu else 'control',
                          input_sha256=sample['input_sha256'], teacher=teacher, positions=positions,
                          history_full=a['output_ids'][div-1], history_phase=b['output_ids'][div-1],
                          prefix=x[:,:-32].contiguous(), tail=x[:,-32:].contiguous(), rope=rope, mask=mask,
                          length=length, states={arm:y for arm in arms}, regions=question_regions(tokenizer,sample),
                          coverage=[]))
    return cases


def pin(keep, votes, region, length):
    # Only fill originally missing protected positions. Evict the weakest
    # remaining standard selections; keep the final 32 and the exact original budget.
    selected = torch.zeros_like(votes, dtype=torch.bool).scatter_(2, keep, True)
    priority = votes.masked_fill(~selected, -torch.inf)
    a, b = region
    priority[:,:,a:b] = torch.inf
    priority[:,:,length-32:] = torch.inf
    return priority.topk(keep.size(-1), dim=-1).indices.sort(dim=-1).values


def dense_decode(layer, hidden, k, v, rope):
    n = hidden.shape[1]
    cache = DynamicCache()
    cache.update(k, v, 0)
    mask = torch.zeros((1, 1, n, k.shape[2]+n), dtype=hidden.dtype, device=hidden.device)
    mask[:,:,:,k.shape[2]:].masked_fill_(torch.ones(n,n,dtype=torch.bool,device=k.device).triu(1),
                                         torch.finfo(hidden.dtype).min)
    return layer(hidden, past_key_value=cache, position_embeddings=rope, attention_mask=mask, use_cache=True)[0]


def replay(layer, case, config, opt, *, coverage=None):
    dense = DynamicCache()
    rp, rt, rd = case['rope']
    case['prefix'] = layer(case['prefix'], past_key_value=dense, position_embeddings=rp, use_cache=True)[0]
    case['tail'], scores, _ = layer(case['tail'], past_key_value=dense, position_embeddings=rt,
                                    attention_mask=case['mask'], output_attentions=True, use_cache=True)
    length = case['length']
    keep = select_prompt_indices([scores], prompt_len=length, model_config=config, prefill_keep_ratio=.7,
                                 sink_tokens=0, obs_window=32, pool_kernel=7, pooling='maxpool')[0]
    indices = {'keep70_fp16':keep, 'keep70_4bit':keep}
    if coverage is not None:
        indices['coverage_packed'] = coverage.apply(keep, scores, repair=True)
    if PIN[0] in case['states']:
        votes = hetero_cache.observation_votes(scores,config.num_key_value_heads,32)
        votes = torch.cat((hetero_cache.pool_scores(votes[:,:,:-32],7,'maxpool'),votes[:,:,-32:]),dim=-1)
        for arm,region in zip(PIN,case['regions']):
            indices[arm] = pin(keep,votes,region,length)
        case['coverage'].append([float(((keep>=a)&(keep<b)).sum().item())/(keep.shape[1]*(b-a))
                                 for a,b in case['regions']])
    del scores
    for arm, hidden in case['states'].items():
        if arm in ('keep70_packed','coverage_packed'):
            cache = PhaseKVCache.from_dense_prefill(dense, prompt_len=length, keep_indices=[indices.get(arm,keep)],
                    prompt_bits=4,cold_bits=4,group_size=32,recent_window=128,lazy_g=16,
                    decode_keep_ratio=.8,optimizations=opt)
            original = layer.self_attn
            layer.self_attn = PhaseKVLlamaAttention(original)
            try:
                updated = []
                for step in range(hidden.shape[1]):
                    updated.append(layer(hidden[:,step:step+1],past_key_value=cache,use_cache=True,
                                         position_embeddings=tuple(x[:,step:step+1] for x in rd))[0])
                    cache.finish_decode_step()
                assert not cache.layers[0].cold
                case['states'][arm] = torch.cat(updated,dim=1)
            finally:
                layer.self_attn = original
            continue
        k, v = dense.key_cache[0], dense.value_cache[0]
        if arm in indices:
            pick = indices[arm].unsqueeze(-1).expand(-1,-1,-1,k.shape[-1])
            k, v = k.gather(2,pick), v.gather(2,pick)
        if '4bit' in arm:
            k = unpack_dequantize(quantize_pack_fast(k,4,2,32))
            v = unpack_dequantize(quantize_pack_fast(v,4,3,32))
        case['states'][arm] = dense_decode(layer, hidden, k, v, rd)


def readout(case, norm, head, tokenizer):
    row = {key:case[key] for key in ('id','cohort','length','input_sha256','teacher','positions',
                                     'history_full','history_phase','regions')}
    if case['coverage']:
        row['standard_retention_in_target_and_control'] = [statistics.mean(v) for v in zip(*case['coverage'])]
    row['predictions'] = {}
    for arm, hidden in case['states'].items():
        logits = torch.nn.functional.linear(norm(hidden[:,[n-1 for n in case['positions']]]),head).float()[0]
        assert torch.isfinite(logits).all()
        predictions = []
        for scores in logits:
            top = scores.argmax().item()
            label = case['teacher'][0]
            full_next = case['history_full']
            predictions.append(dict(top1=top, text=tokenizer.decode([top]), answer_token=label,
                                    label_logit=scores[label].item(), full_next_logit=scores[full_next].item(),
                                    label_minus_full_next=(scores[label]-scores[full_next]).item(),
                                    label_rank=1+int((scores>scores[label]).sum().item())))
        row['predictions'][arm] = predictions
    return row


@torch.inference_mode()
def main():
    samples, phase, full, audit, matched = archive_audit()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'archive_audit.json').write_text(json.dumps(dict(rows=audit, jointly_parsed=matched),indent=2))
    tokenizer = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    config = LlamaConfig.from_pretrained(MODEL,local_files_only=True)
    config._attn_implementation = 'sdpa'
    wm = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        with safe_open(str(MODEL/wm[name]),framework='pt',device='cpu') as handle:
            return handle.get_tensor(name).to(device='cuda',dtype=torch.float16)
    cases = prepare_cases(weight('model.embed_tokens.weight'), LlamaRotaryEmbedding(config=config,device='cuda'),
                          tokenizer,samples,phase,full,audit)
    opt = replace(stage_configs()['p5'],gqa_reuse=True,channel_contiguous_k=True)
    print('Cases by cohort:',dict(collections.Counter(c['cohort'] for c in cases)),flush=True)
    for i in range(config.num_hidden_layers):
        with torch.device('meta'):
            layer = LlamaDecoderLayer(config,layer_idx=0)
        layer.load_state_dict({k:weight(f'model.layers.{i}.{k}') for k in layer.state_dict()},assign=True)
        layer.eval()
        for case in cases:
            replay(layer,case,config,opt)
        del layer
        if (i+1)%4==0:
            print('Replayed layers',i+1,'/',config.num_hidden_layers,flush=True)
    norm = LlamaRMSNorm(config.hidden_size,eps=config.rms_norm_eps).to(device='cuda',dtype=torch.float16)
    norm.weight.copy_(weight('model.norm.weight'))
    head = weight('lm_head.weight')
    rows = [readout(case,norm,head,tokenizer) for case in cases]
    result = dict(model='Llama-3.1-8B-Instruct',layers=config.num_hidden_layers,gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__,mode='fixed teacher prefix, no generated cold blocks, layer streaming, FP16',
                  controls='Shared SDPA/eager-tail prefill. Factorial arms use one dense SDPA backend. 4-bit prompt '
                  'K/V are actually packed then dequantized to FP16. keep70_packed uses the production segmented '
                  'attention with each token streamed separately. Pin arms preserve the exact same budget.',
                  cohorts='7 archived keep80 severe loops plus old-only GSM8K/15, 16 MMLU, two descriptive controls.',
                  rows=rows)
    (OUT/'fixed_prefix.json').write_text(json.dumps(result,indent=2))
    print('FINISHED',OUT,flush=True)


if __name__=='__main__':
    main()
