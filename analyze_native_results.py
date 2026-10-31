import collections as C
import gzip, hashlib, html, json, re, math, statistics as S
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
SRC = ROOT/'不同算法的不同数据集测试/native_llama_20261029'
OUT = SRC.parent/'analysis_20261031'
OUT.mkdir(exist_ok=True)
FONT = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(), 'axes.unicode_minus':False,
    'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
    'figure.facecolor':'white', 'axes.titleweight':'bold', 'savefig.dpi':180})
METHODS=['full','kivi','snapkv','scope','phasekv']
ALL=METHODS+['scope_slide','scope_adaptive']
LABEL=dict(zip(ALL,['Full','KIVI','SnapKV','SCOPE','PhaseKV','SCOPE-Slide','SCOPE-Adaptive']))
COL=dict(zip(ALL,['#64748b','#2563eb','#e6a126','#8b5cf6','#dc4446','#a78bfa','#c4b5fd']))
rows=[json.loads(l) for l in (SRC/'records.jsonl').open(encoding='utf-8')]
manifest=json.loads((SRC/'run_manifest.json').read_text(encoding='utf-8'))
frozenpath=ROOT/'official_llama/bench_assets/frozen/samples.jsonl.gz'
frozen={r['id']:r for r in map(json.loads,gzip.open(frozenpath,'rt',encoding='utf-8'))}
key=lambda r:(r['method'],r['id'],r['repeat'])
actual={key(r):r for r in rows}
expected={(m,s['id'],rep) for s in frozen.values() for m in s['methods'] for rep in range(3 if s['suite']=='performance' else 1)}
assert len(actual)==len(rows)==5861 and set(actual)==expected
raw=[json.loads(p.read_text(encoding='utf-8')) for p in (SRC/'records').rglob('*.json')]
assert len(raw)==len(rows) and all(actual[key(r)]==r for r in raw)
assert hashlib.sha256(frozenpath.read_bytes()).hexdigest()==manifest['protocol']['frozen_data']
assert all(r['input_sha256']==frozen[r['id']]['input_sha256'] for r in rows)
assert all(r['protocol_hash']==manifest['protocol_hash'] for r in rows)
assert all(r['dtype']=='float16' for r in rows)
assert all(len(r['output_ids'])==r['output_tokens'] for r in rows if r['status']=='ok')
assert all(r['output_tokens']==frozen[r['id']]['max_new_tokens'] for r in rows if r['status']=='ok' and r['suite']=='performance')
local_hashes={name:hashlib.sha256((ROOT/'official_llama'/name).read_bytes()).hexdigest()==sha for name,sha in manifest['protocol']['code'].items()}
audit={'records':len(rows),'status':dict(C.Counter(r['status'] for r in rows)),
       'raw_records_equal_export':True,'expected_ids_equal':True,'frozen_hash_matches':True,
       'input_hashes_match':True,'protocol_hash':manifest['protocol_hash'],
       'local_source_matches':local_hashes}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
def select(**kw): return [r for r in rows if all(r.get(k)==v for k,v in kw.items())]
def mean(v): return S.mean(v) if v else float('nan')
def med(v): return S.median(v) if v else float('nan')
def clean(v):
    if isinstance(v,float) and not math.isfinite(v): return None
    if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list): return [clean(x) for x in v]
    return v
def save(name,title,note):
    plt.gcf().suptitle(title,fontsize=17,y=.995)
    plt.gcf().text(.02,.012,note,fontsize=9,color='#475569')
    plt.tight_layout(rect=(0,.045,1,.96))
    plt.savefig(OUT/(name+'.png')); plt.savefig(OUT/(name+'.svg')); plt.close()
    figures.append((name,title,note))
figures=[]
status=[]
for m in ALL:
    v=select(method=m); c=C.Counter(r['status'] for r in v)
    status.append([LABEL[m],len(v),c['ok'],c['oom'],f"{c['ok']/len(v):.1%}"])
fig,ax=plt.subplots(figsize=(11,5))
x=np.arange(len(ALL))
ok=np.array([int(v[2]) for v in status]); oom=np.array([int(v[3]) for v in status]); total=ok+oom
ax.bar(x,100*ok/total,color=[COL[m] for m in ALL]); ax.bar(x,100*oom/total,bottom=100*ok/total,color='#e2e8f0',hatch='///')
for i in x: ax.text(i,102,f'{ok[i]}/{total[i]}',ha='center',fontsize=10)
ax.set(xticks=x,xticklabels=[LABEL[m] for m in ALL],ylim=(0,114),ylabel='完成率 (%)')
save('01_completion','完成情况：5021 成功 / 840 OOM','柱顶为成功数/计划数；灰色为 OOM。两个 SCOPE 变体仅测 LongGenBench，不能直接比较总体完成率。')

# Task/length tables preserve failure denominators and split labels.
groups=C.defaultdict(list)
for r in rows:
    if r['suite']!='performance': groups[(r['suite'],r['task'],r.get('evaluation_split') or 'new_frozen',r['length_budget'],r['method'])].append(r)
quality=[]
for (suite,task,split,budget,m),v in groups.items():
    success=[r for r in v if r['status']=='ok']
    quality.append(dict(suite=suite,task=task,split=split,budget=budget,method=m,n=len(v),ok=len(success),
                        success_score=mean([r['score'] for r in success]),system_score=sum(r.get('score',0) for r in v)/len(v)))
(OUT/'quality_tables.json').write_text(json.dumps(clean(quality),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def quality_panels(suite,filename,title,ncols):
    panels=list(dict.fromkeys((r['task'],r.get('evaluation_split') or 'new_frozen') for r in rows if r['suite']==suite))
    fig,axes=plt.subplots(int(np.ceil(len(panels)/ncols)),ncols,figsize=(15,3.4*int(np.ceil(len(panels)/ncols))),squeeze=False)
    for ax,(task,split) in zip(axes.flat,panels):
        budgets=sorted({q['budget'] for q in quality if q['suite']==suite and q['task']==task and q['split']==split})
        for j,m in enumerate(METHODS):
            qs=[next(q for q in quality if (q['suite'],q['task'],q['split'],q['budget'],q['method'])==(suite,task,split,b,m)) for b in budgets]
            pos=np.arange(len(budgets))+(j-2)*.15
            ax.bar(pos,[q['system_score'] for q in qs],width=.14,color=COL[m],label=LABEL[m])
            for px,q in zip(pos,qs):
                if q['ok']<q['n']: ax.text(px,q['system_score']+2,f"{q['ok']}/{q['n']}",rotation=90,ha='center',fontsize=7)
        metric='ROUGE-L' if task in ['gov_report','qmsum','multi_news'] else 'Code similarity' if task in ['lcc','repobench-p'] else 'QA F1' if suite=='longbench' else 'String match'
        tag={'heldout':'未见','historical_retest':'历史复测','new_frozen':'冻结样本'}[split]
        ax.set(title=f'{task} · {tag}\n{metric}',xticks=np.arange(len(budgets)),xticklabels=[str(b) for b in budgets],ylim=(0,110))
        ax.grid(axis='y',alpha=.15); ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(panels):]: ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.962),ncol=5,frameon=False)
    save(filename,title,'分数 0–100，各任务指标不同；OOM 记 0，分母为全部计划样本。失败组标注成功数/计划数。横轴为长度预算。')
quality_panels('longbench','02_longbench','LongBench：按任务、长度及历史划分比较',3)
quality_panels('ruler','03_ruler','RULER：检索与追踪能力',2)

fig,axes=plt.subplots(1,3,figsize=(15,5))
lgb=[]
for ax,task in zip(axes,['gsm8k','mmlu','csqa']):
    for j,m in enumerate(ALL):
        v=select(suite='longgenbench',task=task,method=m); good=[r for r in v if r['status']=='ok']
        official=sum(r.get('score',0) for r in v)/len(v)
        indexed=sum(r.get('indexed_accuracy_diagnostic',0) for r in v)/len(v)
        comp=sum(r.get('completion_pct',0) for r in v)/len(v)
        lgb.append([task,LABEL[m],len(good),official,indexed,comp,sum(r.get('hit_output_limit',False) for r in good),len(v)])
        ax.bar(j,official,color=COL[m]); ax.plot(j,indexed,'ko',ms=4)
        ax.text(j,official+2,f'{official:.1f}',ha='center',fontsize=8)
    ax.set(title=task.upper(),xticks=range(7),xticklabels=[LABEL[m] for m in ALL],ylim=(0,100))
    ax.tick_params(axis='x',rotation=65)
save('04_longgenbench','LongGenBench：长输出质量明显分化','柱为官方得分，黑点为按 Answer 编号对齐的辅助诊断；均以全部 16 条为分母，OOM 记 0。')

shapes=list(dict.fromkeys(r['task'] for r in rows if r['suite']=='performance'))
perf={}
for shape in shapes:
    for m in METHODS:
        v=select(suite='performance',task=shape,method=m,status='ok')
        by=C.defaultdict(list)
        for r in v: by[r['id']].append(r)
        getters={'speed':lambda r:r['decode_tokens_per_s'],'ttft':lambda r:r['ttft_s'],
                 'kv':lambda r:r['final_storage']['physical_bytes']/2**20,
                 'owned':lambda r:r['final_storage']['total_owned_bytes']/2**20,
                 'peak':lambda r:r['peak_allocated_bytes']/2**30}
        a={'ok':len(v)}
        for f,get in getters.items():
            vals=[med([get(r) for r in vv]) for vv in by.values()]
            a[f]=med(vals); a[f+'_min']=min(vals) if vals else float('nan'); a[f+'_max']=max(vals) if vals else float('nan')
        perf[shape,m]=a
(OUT/'performance_tables.json').write_text(json.dumps(clean([dict(shape=s,method=m,**a) for (s,m),a in perf.items()]),indent=2,allow_nan=False),encoding='utf-8')
fig,axes=plt.subplots(2,2,figsize=(16,10))
for ax,(f,title) in zip(axes.flat,[('speed','解码速度 (token/s) ↑'),('ttft','首 token 延迟，含压缩 (s) ↓'),('kv','最终持久 KV 分配容量 (MiB) ↓'),('peak','整个进程 PyTorch 分配峰值 (GiB) ↓')]):
    for m in METHODS:
        vals=[perf[s,m][f] for s in shapes]
        ax.plot(range(len(shapes)),vals,'o-',color=COL[m],label=LABEL[m],ms=4)
        ax.vlines(range(len(shapes)),[perf[s,m][f+'_min'] for s in shapes],[perf[s,m][f+'_max'] for s in shapes],color=COL[m],alpha=.6)
    ax.set(title=title,xticks=range(len(shapes)),xticklabels=shapes)
    ax.tick_params(axis='x',rotation=40); ax.grid(alpha=.15)
    if f in ['ttft','kv']: ax.set_yscale('log')
axes[0,0].legend(ncol=3,fontsize=9)
save('05_performance','固定长度性能与空间：相同输入/输出形状','每 prompt 先取 3 次中位数，再取 3 个 prompt 的中位数；细线为 prompt 中位数范围。断点为 OOM；末项为 32K 扩展。')

fig,ax=plt.subplots(figsize=(13,5))
matrix=np.array([[perf[s,m]['ok'] for s in shapes] for m in METHODS])
ax.imshow(matrix,vmin=0,vmax=9,cmap='Blues',aspect='auto')
for i in range(5):
    for j in range(9): ax.text(j,i,'9/9' if matrix[i,j] else 'OOM\n0/9',ha='center',va='center',color='white' if matrix[i,j] else '#9f1239')
ax.set(yticks=range(5),yticklabels=[LABEL[m] for m in METHODS],xticks=range(9),xticklabels=shapes)
save('06_capacity','性能形状的完成情况','全部性能 OOM 均在预热存储诊断发生；0/9 表示正式测量未产生成功记录，不代表 9 次独立计时运行均 OOM。')

paired=json.loads((SRC/'quality_paired_vs_full.json').read_text(encoding='utf-8'))
pv=[r for r in paired if r['method']=='phasekv']
fig,ax=plt.subplots(figsize=(12,10))
for i,r in enumerate(pv):
    a,b=r['bootstrap95_low'],r['bootstrap95_high']; d=r['delta_vs_full_points']
    color='#dc4446' if b<0 else '#2563eb'
    ax.plot([a,b],[i,i],color=color,lw=2); ax.plot(d,i,'o',color=color)
    ax.text(max(b,0)+1,i,f'{d:+.2f} [{a:.2f}, {b:.2f}]',va='center',fontsize=8)
ax.axvline(0,color='#94a3b8',ls='--')
ax.set(yticks=range(len(pv)),yticklabels=[f"{r['task']} / {r['evaluation_split']} (n={r['paired_document_clusters']})" for r in pv],xlabel='PhaseKV − Full（分数点）',xlim=(-70,32))
ax.invert_yaxis()
save('07_paired_quality','PhaseKV 相对 Full：配对质量差异及 95% 区间','沿用运行器按 cluster_id 聚类的 2000 次 bootstrap；仅成功配对，未做多重比较校正。RULER 的跨长度聚类是运行器的设定。')

# A second memory view makes PhaseKV auxiliary allocations visible.
fig,ax=plt.subplots(figsize=(14,5))
for j,m in enumerate(METHODS):
    pos=np.arange(9)+(j-2)*.15
    base=np.array([perf[s,m]['kv'] for s in shapes]); owned=np.array([perf[s,m]['owned'] for s in shapes])
    ax.bar(pos,base,.14,color=COL[m],label=LABEL[m]); ax.bar(pos,owned-base,.14,bottom=base,color=COL[m],hatch='///',alpha=.45)
ax.set(xticks=range(9),xticklabels=shapes,ylabel='最终存储 (MiB)'); ax.tick_params(axis='x',rotation=30); ax.legend(ncol=5)
save('08_owned_memory','KV 本体与辅助空间','实色为 physical_bytes，斜线为额外辅助空间；缺柱为 OOM。此图为最终存储，不是全过程峰值。')

def table(headers,data):
    def fmt(v): return ('—' if not math.isfinite(v) else f'{v:.2f}') if isinstance(v,float) else str(v)
    return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(v))+'</td>' for v in row)+'</tr>' for row in data)+'</tbody></table>'
common=[s for s in shapes if perf[s,'full']['ok'] and perf[s,'phasekv']['ok']]
ratios=[perf[s,'phasekv']['speed']/perf[s,'full']['speed'] for s in common]
mem=[1-perf[s,'phasekv']['kv']/perf[s,'full']['kv'] for s in common]
owned=[1-perf[s,'phasekv']['owned']/perf[s,'full']['owned'] for s in common]
vs_kivi=[perf[s,'phasekv']['speed']/perf[s,'kivi']['speed'] for s in shapes if perf[s,'phasekv']['ok']]
notes=[
 '本轮 5861 条计划记录全部有结果：5021 成功、840 OOM（14.33%），没有 error 或 pending。原子 records 与 records.jsonl 内容一致，冻结输入哈希、逐条输入哈希、协议哈希及固定输出长度核验通过。',
 f'PhaseKV 完成全部 1072 条质量生成及 72/81 条性能测量；仅 32000+256 性能形状失败。KIVI 全部完成。Full 在 16384+256、16384+4096、32000+256 的性能预热失败。',
 f'在与 Full 共同成功的 {len(common)} 个性能形状上，PhaseKV 解码速度为 Full 的 {min(ratios):.1%}–{max(ratios):.1%}，因此本轮不支持“比 Full 更快”。相对 KIVI，PhaseKV 在共同成功的 8 个形状上快 {min(vs_kivi):.2f}–{max(vs_kivi):.2f} 倍。',
 f'共同成功形状中，PhaseKV 最终 KV 本体容量比 Full 减少 {min(mem):.1%}–{max(mem):.1%}；把辅助空间也计入后，减少 {min(owned):.1%}–{max(owned):.1%}。这些是最终分配容量，不是全过程峰值。',
 'LongBench 多数任务与 Full 的配对差异较小；gov_report 下降 1.70 分，未校正 95% 区间为 [-2.78, -0.61]。其余区间包含或触及 0，不能据此宣称统计等效。narrativeqa 全为历史复测；qmsum 分为未见和历史复测。',
 'RULER 中 PhaseKV 与 Full 的任务级平均分接近：multikey 与 vt 均为 100；multivalue 均为 98.70；cwe 为 88.65 对 88.54。SCOPE 大量长输入 OOM，必须同时看失败分母。',
 'LongGenBench 暴露明显退化：PhaseKV 的 GSM8K/MMLU/CSQA 官方得分为 20.63/35.63/24.53，Full 为 53.54/48.13/75.94；配对差值分别为 -32.92/-12.50/-51.41 分，三个未校正区间均低于 0。',
 'PhaseKV 的 GSM8K/MMLU/CSQA 按编号解析完成率为 62.92%/73.54%/55.63%；CSQA 全部 16 条触及 4096 输出上限。长输出问题不仅是官方 parser 的差异，按编号辅助得分同样下降。完成率指成功解析的答案比例，不能把所有未解析项都认定为模型未作答。',
 'SCOPE 主版本 762 条 OOM，约占全部 OOM 的 90.7%。代表性栈显示 SCOPE 在 H2O prefill 完整注意力矩阵分配失败；Full/SnapKV 在 KV 拼接处失败；PhaseKV 的 32K 失败栈位于 MLP。这些定位描述失败位置，尚不足以证明唯一根因。',
 '性能测试的 135 条 OOM 记录全来自预热存储诊断。成功诊断的开销不进入速度表，但预热失败会阻止正式测量；容量结果仅适用于本轮整个执行协议，不能视作脱离 runner 的纯推理容量上限。',
 'Full 的 KV 为 8 头布局，SnapKV/SCOPE 为 32 头布局。4096+256 形状最终 KV：Full 543.88 MiB，SnapKV/SCOPE 1151.50 MiB，PhaseKV 152.06 MiB（含辅助为 184.93 MiB）。不能用保留 token 数代替物理字节。',
 '建议优先核查 PhaseKV 长输出的答案序列、输出上限与压缩影响，再安排独立的参数或 P5 消融实验；本轮数据不能单独定位到某个量化模块。新增实验必须新开输出目录，保持本轮证据。',
 '统计限制：各任务指标不同，不合成为总准确率。质量生成的自然 EOS 速度不用于性能比较。性能先每 prompt 取重复中位数，再跨 prompt 取中位数；图中范围不是置信区间。不同方法按阶段分批运行，温度/频率仅有端点采样。',
 'narrativeqa 的 32 个选择索引实际对应 19 个独立上下文簇；qmsum 的 27 个历史索引对应 19 个上下文簇。配对统计使用上下文簇，避免把共享上下文重复当作独立证据。每组 bootstrap 区间未校正多重比较。',
 '源码审计：'+ ('清单中的 9 个本地文件哈希全部与运行协议一致。' if all(local_hashes.values()) else '本地文件与运行清单存在差异，详见 audit.json。')+'这里只核对清单指纹，未重新执行 GPU 推理或重算全部官方评分。'
]
repetition=[]
for r in rows:
    if r['suite']!='longgenbench' or r['status']!='ok': continue
    counts=C.Counter(re.findall(r'Answer_(\d+)\s*:',r['prediction']))
    if counts and max(counts.values())>=5:
        repetition.append(dict(method=r['method'],task=r['task'],id=r['id'],score=r['score'],
                               most_common_headers=counts.most_common(3),tail=r['prediction'][-800:]))
(OUT/'repeated_answer_examples.json').write_text(json.dumps(repetition,ensure_ascii=False,indent=2),encoding='utf-8')
repcounts=C.Counter((r['method'],r['task']) for r in repetition)
notes.insert(8,'输出原文复核：以“同一个 Answer 编号出现至少 5 次”为可复现诊断，PhaseKV 的 CSQA 有 '+str(repcounts['phasekv','csqa'])+'/16 条触发，Full 为 '+str(repcounts['full','csqa'])+'/16 条。比如 longgenbench/csqa/23 和 /26 均反复输出 Answer_6:，官方分与按编号完成率均为 0，输出达到 4096 token。另有 /25 在完成目标答案后继续生成编号和重复内容。原文尾部及各方法触发列表保存在 repeated_answer_examples.json。该规则只检查重复编号，不能覆盖所有重复退化。')
summary=json.loads((SRC/'summary.json').read_text(encoding='utf-8'))
for q in quality:
    old=next(g for g in summary['groups'] if (g['suite'],g['task'],g['evaluation_split'],g['budget'],g['method'])==(q['suite'],q['task'],q['split'],str(q['budget']),q['method']))
    assert old['ok']==q['ok'] and old['expected']==q['n']
    assert abs(old['score_system_failures_zero']-q['system_score'])<1e-9
perfrows=[[s,LABEL[m],perf[s,m]['ok'],*(perf[s,m][f] if perf[s,m]['ok'] else 'OOM' for f in ['speed','ttft','kv','owned','peak'])] for s in shapes for m in METHODS]
qrows=[[q[k] for k in ['suite','task','split','budget','method','ok','n','success_score','system_score']] for q in quality]
body='<h1>原生 Llama 正式实验分析</h1><p>2026-10-31 · Llama-3.1-8B-Instruct · FP16 · RTX 4090 D · batch=1</p>'
body+='<h2>主要发现与解释边界</h2>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes)
body+='<h2>完成情况</h2>'+table(['方法','计划','成功','OOM','成功率'],status)
for name,title,note in figures: body+=f'<section><h2>{title}</h2><p>{note}</p><a href="{name}.svg">下载 SVG</a><img src="{name}.png" alt="{title}"></section>'
body+='<h2>LongGenBench 明细</h2>'+table(['任务','方法','成功','官方系统分','编号诊断分','解析完成率%','触及输出上限','计划'],lgb)
body+='<h2>固定长度性能明细</h2>'+table(['输入+输出','方法','成功/9','token/s','TTFT秒','KV MiB','含辅助 MiB','分配峰值 GiB'],perfrows)
body+='<h2>质量明细（成功均值与失败记零并列）</h2>'+table(['套件','任务','划分','长度','方法','成功','计划','成功均分','失败记零系统分'],qrows)
body+='<h2>来源与复现</h2><p>源目录：'+html.escape(str(SRC))+'</p><p>protocol_hash: '+manifest['protocol_hash']+'</p><p>audit.json 保存一致性核验；quality_tables.json / performance_tables.json 保存图表数据。所有原始文件保持原样。重现：项目 .venv/Scripts/python.exe analyze_native_results.py。</p>'
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PhaseKV 正式实验分析</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#18243a;max-width:1380px;margin:40px auto;padding:0 28px}h1{font-size:34px}h2{margin-top:38px;color:#253f68}p{max-width:1150px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{border-bottom:1px solid #e2e8f0;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f8}tr:nth-child(even){background:#f8fafc}section{page-break-inside:avoid}a{color:#2563eb}</style>'+body+'</html>'
(OUT/'分析报告.html').write_text(doc,encoding='utf-8')
(OUT/'分析结论.txt').write_text('\n\n'.join(notes),encoding='utf-8')
print(json.dumps({'output':str(OUT),'audit':audit,'phase_speed_ratio_full':ratios,'phase_speed_ratio_kivi':vs_kivi,'kv_saving':mem,'owned_saving':owned,'figures':len(figures)},ensure_ascii=True,indent=2))
