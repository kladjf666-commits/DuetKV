"""Evaluate a prespecified all-question coverage floor at the first Full/PhaseKV fork."""
import collections
import json
import sys
from pathlib import Path
from dataclasses import replace

import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

import probe_longgen_quality as probe
from PhaseKV.optimizations import stage_configs
sys.path.insert(0, str(probe.ROOT / 'workspace_migration/quality_coverage'))
from coverage_policy import CoveragePlan, question_units


@torch.inference_mode()
def main():
    samples, phase, full, archive, _ = probe.archive_audit()
    tokenizer = Tokenizer.from_file(str(probe.MODEL/'tokenizer.json'))
    config = LlamaConfig.from_pretrained(probe.MODEL, local_files_only=True)
    config._attn_implementation = 'sdpa'
    weights = json.loads((probe.MODEL/'model.safetensors.index.json').read_text())['weight_map']
    def weight(name):
        with safe_open(str(probe.MODEL/weights[name]),framework='pt',device='cpu') as handle:
            return handle.get_tensor(name).to(device='cuda',dtype=torch.float16)
    cases = probe.prepare_cases(weight('model.embed_tokens.weight'), LlamaRotaryEmbedding(config=config,device='cuda'),
                                tokenizer,samples,phase,full,archive,all_cases=True)
    plans = {}
    for case in cases:
        s = samples[case['id']]
        units = question_units(tokenizer,s['input_ids'],s['task'])
        if sum(u['kind']=='stem' for u in units) != len(s['answers']):
            raise ValueError(f'Question count mismatch: {s["id"]}')
        initial = case['states']['keep70_packed']
        case['states'] = {arm:initial for arm in ('keep70_packed','coverage_packed')}
        plans[s['id']] = CoveragePlan(units,case['length'],'cuda')
    opt = replace(stage_configs()['p5'],gqa_reuse=True,channel_contiguous_k=True)
    for i in range(config.num_hidden_layers):
        with torch.device('meta'):
            layer = LlamaDecoderLayer(config,layer_idx=0)
        layer.load_state_dict({k:weight(f'model.layers.{i}.{k}') for k in layer.state_dict()},assign=True)
        layer.eval()
        for case in cases:
            probe.replay(layer,case,config,opt,coverage=plans[case['id']])
        del layer
        if (i+1)%4==0:
            print('Replayed',i+1,'/ 32 layers for',len(cases),'prompts',flush=True)
    norm = LlamaRMSNorm(config.hidden_size,eps=config.rms_norm_eps).to(device='cuda',dtype=torch.float16)
    norm.weight.copy_(weight('model.norm.weight'))
    head = weight('lm_head.weight')
    rows = []
    for case in cases:
        row = probe.readout(case,norm,head,tokenizer)
        plan = plans[case['id']]
        if len(plan.layers) != 32:
            raise RuntimeError('Incomplete per-layer audit')
        row['selection'] = plan.totals()
        row['per_head_budget'] = plan.layers[0]['budget']
        row['verified_greedy_prefix'] = {}
        initial = int(torch.nn.functional.linear(norm(case['tail'][:,-1]),head).argmax(-1).item())
        teacher = case['teacher']
        for arm,hidden in case['states'].items():
            next_ids = torch.nn.functional.linear(norm(hidden),head).argmax(-1)[0].tolist()
            predictions = [initial,*next_ids]
            common = next((i for i,(a,b) in enumerate(zip(predictions,teacher)) if a!=b), len(teacher))
            row['verified_greedy_prefix'][arm] = predictions[:common+1]
        rows.append(row)
    totals = collections.Counter()
    for row in rows:
        totals.update(row['selection'])
    assert len(rows)==48 and totals['after_stem']==totals['after_option']==0
    out = probe.ROOT/'results/longgen_question_coverage'
    out.mkdir(exist_ok=True)
    result = dict(model='Llama-3.1-8B-Instruct', gpu=torch.cuda.get_device_name(), layers=32,
                  mode='Fixed shared teacher prefixes; packed FP16-hot/4-bit prompt paths. '
                       'verified_greedy_prefix ends at the first disagreement with the teacher, inclusive.',
                  policy='Round(0.7*N) per layer/head, final 32 unchanged. '
                         'Use min(n,max(6,ceil(2*n/3))) stem content and min(n,max(3,ceil(2*n/3))) per visible option.',
                  totals=dict(totals), rows=rows)
    (out/'fixed_prefix.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('FINISHED',out/'fixed_prefix.json',flush=True)


if __name__=='__main__':
    main()
