"""Profile-based comparison of six real-KV methods on the same model/inputs."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from models.registry import FAMILIES, load_profile, profile_names, integration, validate_methods, validate_model_config, estimated_parameter_count
from eval.real_kv_policy import forbid_fake_quantization
from models.generation import stop_token_ids
from eval import metrics
from scripts import run_longbench as lb

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def checkpoint_status(profile, model_path=None):
    try:
        folder = Path(model_path) if model_path else Path(snapshot_download(
            profile['model_id'], revision=profile['revision'], local_files_only=True))
        if (folder/'model.safetensors.index.json').is_file():
            index=json.loads((folder/'model.safetensors.index.json').read_text())
            shards=set(index['weight_map'].values())
            available=all((folder/s).is_file() for s in shards)
        else:
            available=(folder/'model.safetensors').is_file()
        return {'path':str(folder),'weights_present':available}
    except (OSError, ValueError):
        return {'path':model_path,'weights_present':False}


def preflight(profile, methods, model_path=None, allow_download=False):
    import transformers
    weights=2*estimated_parameter_count(profile['architecture'])
    cuda=torch.cuda.is_available()
    total=free=0
    reasons=[]
    if transformers.__version__ != profile['verified_transformers']:
        reasons.append(f"Use validated transformers=={profile['verified_transformers']}; installed {transformers.__version__}")
    if not cuda:
        reasons.append('CUDA is unavailable')
    else:
        free,total=torch.cuda.mem_get_info()
        if not torch.cuda.is_bf16_supported(): reasons.append('BF16 is unsupported by this GPU')
        if free < weights + 2**30:
            reasons.append('Free VRAM is below BF16 weights plus 1 GiB minimum working space')
    checkpoint=checkpoint_status(profile,model_path)
    if not checkpoint['weights_present'] and not allow_download:
        reasons.append('Weights are missing; provide --model-path or explicitly use --allow-download')
    return {'profile':profile['name'],'model_id':profile['model_id'],
            'methods':[m['method'] for m in methods], 'cuda_available':cuda,
            'gpu':torch.cuda.get_device_name() if cuda else None,
            'gpu_total_gib':total/2**30,'gpu_free_gib':free/2**30,
            'bf16_weight_gib':weights/2**30, 'checkpoint':checkpoint,
            'ready':not reasons,'blocking_reasons':reasons,
            'note':'Memory guard is a lower bound, not an OOM guarantee. Eager attention scores grow quadratically with input length.'}


def load_named_model(profile, args):
    source=args.model_path or profile['model_id']
    kwargs={'local_files_only':not args.allow_download,'trust_remote_code':False}
    if not args.model_path: kwargs['revision']=profile['revision']
    config=AutoConfig.from_pretrained(source,**kwargs)
    validate_model_config(profile,config)
    tokenizer=AutoTokenizer.from_pretrained(source,**kwargs)
    if profile['family']=='llama' and not getattr(tokenizer,'chat_template',None):
        raise ValueError('Original Llama chat/instruct tokenizer with chat_template is required')
    model=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.bfloat16,
                attn_implementation='eager',low_cpu_mem_usage=True,**kwargs).to('cuda').eval()
    if getattr(model,'is_quantized',False): raise ValueError('Unexpected model weight quantization')
    devices={str(p.device) for p in model.parameters()}
    if any(not d.startswith('cuda') for d in devices): raise ValueError('CPU/disk weight offload is unsupported')
    if 'expected_eos_token_ids' in profile and stop_token_ids(model,tokenizer)!=set(profile['expected_eos_token_ids']):
        raise ValueError('Llama EOS configuration differs from this profile; provide the original generation_config.json')
    return model,tokenizer


def prepare_samples(profile, args, tokenizer):
    if args.mode=='smoke':
        ids=tokenizer.apply_chat_template([{'role':'user','content':args.prompt}],tokenize=True,
                                         add_generation_prompt=True,return_tensors='pt')
        return [{'sample_id':'smoke:0','task':'smoke','ids':ids,'answers':[],
                 'max_new':args.max_new_tokens or 80}],{}
    lb.LONGBENCH_DIR=str(Path(args.longbench_dir).resolve())
    settings=profile['quality_test']
    tasks=args.tasks.split(',') if args.tasks else settings['tasks']
    if len(tasks)!=len(set(tasks)) or any(t not in lb.DATASET2MAXLEN for t in tasks):
        raise ValueError('Unknown or duplicate LongBench tasks')
    count=args.n_samples or settings['samples_per_task']
    length=args.max_length or settings['max_length_before_chat_template']
    samples=[]; hashes={}
    for task in tasks:
        path=Path(lb.LONGBENCH_DIR)/'data'/f'{task}.jsonl'
        hashes[task]=hashlib.sha256(path.read_bytes()).hexdigest()
        data=lb.load_task(task,count)
        if len(data)!=count: raise ValueError(f'Insufficient {task} samples')
        for i,item in enumerate(data):
            samples.append({'sample_id':f'{task}:{i}','task':task,
                'ids':getattr(integration(profile),'build_input_ids',lb.build_input_ids)(tokenizer,task,item,length),'answers':item['answers'],
                'max_new':args.max_new_tokens or lb.DATASET2MAXLEN[task]})
    return samples,hashes


def summary(records,methods):
    refs={r['sample_id']:r for r in records if r['method']=='full'}
    rows=[]
    for spec in methods:
        selected=[r for r in records if r['method']==spec['method']]
        if not selected: continue
        tasks=sorted({r['task'] for r in selected})
        scores={t:statistics.mean(r['qa_f1'] for r in selected if r['task']==t)*100 for t in tasks
                if any(r['task']==t and r['qa_f1'] is not None for r in selected)}
        total_ms=sum(r['wall_ms'] for r in selected)
        rows.append({'method':spec['method'],'params':{k:v for k,v in spec.items() if k!='method'},
            'samples':len(selected),'qa_f1':statistics.mean(scores.values()) if scores else None,
            'per_task':scores,'mean_physical_kv_mib':statistics.mean(r['physical_kv_bytes'] for r in selected)/2**20,
            'mean_physical_kv_saving_pct':statistics.mean(r['physical_kv_saving_pct'] for r in selected),
            'triggered_samples':sum(r['compression_trace']['status']=='triggered' for r in selected),
            'decode_triggered_steps':sum(r['compression_trace']['summary']['decode']['triggered_steps'] for r in selected),
            'trace_status_counts':dict(Counter(r['compression_trace']['status'] for r in selected)),
            'mean_wall_ms':total_ms/len(selected),
            'end_to_end_tokens_per_second':sum(len(r['generated_ids']) for r in selected)/(total_ms/1000),
            'peak_allocated_mib':max(r['peak_gpu_bytes'] for r in selected)/2**20,
            'exact_matches_vs_full':sum(r['generated_ids']==refs[r['sample_id']]['generated_ids'] for r in selected if r['sample_id'] in refs),
            'full_comparison_samples':sum(r['sample_id'] in refs for r in selected)})
    return rows


def save_report(prefix,meta,records,methods,status,error=None):
    rows=summary(records,methods)
    data={'meta':{**meta,'status':status,'error':error},'rows':rows}
    prefix.with_suffix('.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=[f"# {meta['profile']['name']} KV 对比",'',f"状态：{status}；模式：{meta['mode']}。",'',
           '同一次运行共用一份BF16模型和相同输入。KIVI/PhaseKV使用真实打包；其余压缩方法为token淘汰。',
           'SCOPE仅为ALLKV+Slide的GQA适配。各方法默认参数不同，本表不是相同KV内存预算对比。',
           '速度为端到端输出tokens/s，包含prefill、压缩维护和触发日志，不代表纯decode速度；smoke不计质量分数。','',
           '| 方法 | 样本 | QA F1 | KV MiB | KV节省% | 触发样本 | decode触发步 | 端到端tokens/s |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        score='—' if r['qa_f1'] is None else f"{r['qa_f1']:.4f}"
        lines.append(f"| {r['method']} | {r['samples']} | {score} | {r['mean_physical_kv_mib']:.4f} | {r['mean_physical_kv_saving_pct']:.4f} | {r['triggered_samples']} | {r['decode_triggered_steps']} | {r['end_to_end_tokens_per_second']:.2f} |")
    if error: lines.extend(['',f'失败原因：{error}'])
    prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(argv=None,default_profile=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=profile_names(),default=default_profile,required=default_profile is None)
    p.add_argument('--mode',choices=('check','smoke','longbench'),default='check')
    p.add_argument('--model-path',help='Local complete checkpoint; architecture must match the named profile')
    p.add_argument('--allow-download',action='store_true')
    p.add_argument('--methods',help='Comma-separated subset of methods')
    p.add_argument('--compression-config',type=Path,help='JSON containing a methods list')
    p.add_argument('--longbench-dir',default=str(ROOT/'.cache/longbench'))
    p.add_argument('--tasks')
    p.add_argument('--n-samples',type=int)
    p.add_argument('--max-length',type=int)
    p.add_argument('--max-new-tokens',type=int)
    p.add_argument('--fixed-output',action='store_true',help='Smoke only: ignore EOS to exercise maintenance thresholds')
    p.add_argument('--prompt',default='Explain the purpose, memory cost, and limitations of the key-value cache in a language model. Include several examples.')
    p.add_argument('--out',type=Path,help='New result prefix, not an existing run')
    args=p.parse_args(argv)
    for name in ('n_samples','max_length','max_new_tokens'):
        if getattr(args,name) is not None and getattr(args,name)<1: p.error(f'{name} must be positive')
    if args.fixed_output and args.mode!='smoke': p.error('--fixed-output is only for functional smoke checks')
    profile=load_profile(args.profile)
    methods=(json.loads(args.compression_config.read_text()) if args.compression_config else profile['compression'])['methods']
    if args.methods:
        selected=args.methods.split(',')
        if len(set(selected))!=len(selected) or set(selected)-{m['method'] for m in methods}: p.error('Unknown or duplicate methods')
        methods=[m for m in methods if m['method'] in selected]
    validate_methods(profile,methods)
    check=preflight(profile,methods,args.model_path,args.allow_download)
    if args.mode=='check':
        print(json.dumps(check,ensure_ascii=False,indent=2))
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            with args.out.with_suffix('.json').open('x') as f: json.dump(check,f,ensure_ascii=False,indent=2)
        return 0 if check['ready'] else 2
    if not check['ready']: raise RuntimeError('; '.join(check['blocking_reasons']))
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    prefix=args.out or ROOT/'results'/profile['name']/f'{args.mode}_{stamp}'
    if any(prefix.with_suffix(s).exists() for s in ('.json','.md','.predictions.jsonl')):
        raise FileExistsError('Use a new result prefix; existing measurements are never overwritten')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(1234); torch.set_num_threads(4)
    forbid_fake_quantization()
    model,tokenizer=load_named_model(profile,args)
    samples,data_hashes=prepare_samples(profile,args,tokenizer)
    for s in samples:
        s['input_hash']=digest(s['ids'].tolist())
        if s['ids'].shape[1]+s['max_new']>model.config.max_position_embeddings:
            raise ValueError(f"Actual input exceeds context budget: {s['sample_id']}")
    meta={'profile':profile,'methods':methods,'mode':args.mode,'seed':1234,
          'model_source':args.model_path or profile['model_id'],
          'resolved_revision':getattr(model.config,'_commit_hash',None),
          'local_model_path_override':bool(args.model_path),
          'checkpoint_identity_note':'All methods use one loaded model. A local override is architecture-checked; its publisher identity is not inferred from its directory name.',
          'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'eager','batch_size':1,
          'torch':torch.__version__,'transformers':__import__('transformers').__version__,
          'stop_on_eos':not args.fixed_output,'data_hashes':data_hashes,
          'input_hashes':{s['sample_id']:s['input_hash'] for s in samples},
          'actual_prompt_tokens':{s['sample_id']:s['ids'].shape[1] for s in samples},
          'generation_limits':{s['sample_id']:s['max_new'] for s in samples}}
    meta['eos_token_ids']=sorted(stop_token_ids(model,tokenizer)) if profile['family']=='llama' else [getattr(tokenizer,'eos_token_id',None)]
    meta['prompt_encoding']='native_chat_template_once' if profile['family']=='llama' else 'legacy_qwen_longbench'
    meta['requested_max_length_before_chat_template']=args.max_length or profile['quality_test']['max_length_before_chat_template']
    meta['requested_samples_per_task']=args.n_samples or profile['quality_test']['samples_per_task']
    source_files=[ROOT/p for p in ('eval/compare_models.py','eval/runtime.py','eval/compression_trace.py',
                  'eval/real_kv_policy.py','eval/compressors/head_ops.py','models/registry.py',
                  'models/packed_runtime.py','models/generation.py')]
    family_folder=FAMILIES[profile['family']].rsplit('.',1)[0].replace('.','/')
    for folder in (family_folder,'KIVI','SnapKV','SCOPE','PyramidKV','PhaseKV','PhaseKV/kernels'):
        source_files.extend((ROOT/folder).glob('*.py'))
    meta['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source_files}
    meta['protocol_hash']=digest(meta)
    records=[]; runner=integration(profile).generate
    save_report(prefix,meta,records,methods,'running')
    try:
        with prefix.with_suffix('.predictions.jsonl').open('x',buffering=1) as handle:
            for spec in methods:
                gc.collect(); torch.cuda.empty_cache()
                runner(model,tokenizer,samples[0]['ids'],spec,max_new_tokens=4)
                for index,s in enumerate(samples):
                    torch.cuda.synchronize(); start=time.perf_counter()
                    gen=runner(model,tokenizer,s['ids'],spec,max_new_tokens=s['max_new'],stop_on_eos=not args.fixed_output)
                    torch.cuda.synchronize(); wall=(time.perf_counter()-start)*1000
                    written=gen['prompt_len']+len(gen['generated_ids'])-1
                    cfg=model.config
                    full_bytes=metrics.kv_cache_dtype_bytes(cfg.num_hidden_layers,cfg.num_key_value_heads,cfg.hidden_size//cfg.num_attention_heads,written)
                    physical=gen['compression_trace']['final_state']['physical_bytes']
                    if spec['method']=='full' and physical!=full_bytes: raise AssertionError('Full cache byte accounting mismatch')
                    record={'protocol_hash':meta['protocol_hash'],'method':spec['method'],'sample_id':s['sample_id'],
                        'task':s['task'],'input_hash':s['input_hash'],'prompt_tokens':gen['prompt_len'],
                        'generated_ids':gen['generated_ids'],'prediction':gen['text'],'answers':s['answers'],
                        'qa_f1':lb.best_qa_f1(gen['text'].strip(),s['answers']) if s['answers'] else None,
                        'written_tokens':written,'physical_kv_bytes':physical,'full_reference_bytes':full_bytes,
                        'physical_kv_saving_pct':100*(1-physical/full_bytes),'wall_ms':wall,
                        'peak_gpu_bytes':gen['peak_gpu_bytes'],'compression_trace':gen['compression_trace']}
                    if 'storage_evidence' in gen: record['storage_evidence']=gen['storage_evidence']
                    handle.write(json.dumps(record,ensure_ascii=False)+'\n'); records.append(record)
                    if (index+1)%16==0 or index+1==len(samples):
                        print(f"{profile['name']} / {spec['method']}: {index+1}/{len(samples)}",flush=True)
                        save_report(prefix,meta,records,methods,'running')
        save_report(prefix,meta,records,methods,'complete')
    except Exception as exc:
        save_report(prefix,meta,records,methods,'failed',f'{type(exc).__name__}: {exc}')
        raise
    print(f'Results: {prefix}.md',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
