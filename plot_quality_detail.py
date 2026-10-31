"""Detailed quality comparison from immutable native Llama benchmark records."""
from pathlib import Path
import collections as C
import json, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'不同算法的不同数据集测试/native_llama_20261029'
OUT=SRC.parent/'analysis_20261031'
rows=[json.loads(l) for l in (SRC/'records.jsonl').open(encoding='utf-8')]
assert len(rows)==5861
methods=['full','kivi','snapkv','scope','phasekv']
extra=['scope_slide','scope_adaptive']
names={'full':'Full','kivi':'KIVI','snapkv':'SnapKV','scope':'SCOPE','phasekv':'PhaseKV','scope_slide':'S-Slide','scope_adaptive':'S-Adaptive'}
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':12,'svg.fonttype':'path'})
groups=C.defaultdict(list)
for r in rows:
    if r['suite']!='performance': groups[(r['suite'],r['task'],r.get('evaluation_split') or 'new_frozen',r['length_budget'],r['method'])].append(r)

def stats(k,field='score'):
    v=groups[k]; good=[r for r in v if r['status']=='ok']
    return {'score':sum(r.get(field,0) for r in v)/len(v),'ok':len(good),'n':len(v),
            'success':sum(r[field] for r in good)/len(good) if good else None}
lbkeys=list(dict.fromkeys(k[:4] for k in groups if k[0]=='longbench'))
rukeys=list(dict.fromkeys(k[:4] for k in groups if k[0]=='ruler'))
lgkeys=list(dict.fromkeys(k[:4] for k in groups if k[0]=='longgenbench'))
assert len(lbkeys)==22 and len(rukeys)==12 and len(lgkeys)==3
qacmap=plt.get_cmap('YlGnBu'); dmap=plt.get_cmap('RdBu'); norm=Normalize(0,100); dn=TwoSlopeNorm(vmin=-55,vcenter=0,vmax=55)
fig=plt.figure(figsize=(26,19),facecolor='#f8fafc')
gs=fig.add_gridspec(2,2,left=.155,right=.985,top=.835,bottom=.12,width_ratios=[1,1.22],height_ratios=[12,9],wspace=.48,hspace=.34)

def heat(ax,keys,labels,ms,title,fields=None,breaks=()):
    if fields is None: fields=['score']*len(keys)
    nc=len(ms)+1
    for i,(k,field) in enumerate(zip(keys,fields)):
        values=[stats((*k,m),field) for m in ms]
        best=max(v['score'] for v in values)
        for j,v in enumerate(values):
            bg=qacmap(norm(v['score']))
            if v['ok']==0: bg='#e2e8f0'
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=bg,edgecolor='white',lw=1.5))
            if v['ok']==0:
                text='OOM\n0/'+str(v['n']); color='#9f1239'; size=10
            elif v['ok']<v['n']:
                text=f"{v['score']:.1f}\n{v['ok']}/{v['n']}成功"; color='white' if v['score']>65 else '#172033';size=10
            else:
                text=f"{v['score']:.1f}";color='white' if v['score']>65 else '#172033';size=12
            ax.text(j,i,text,ha='center',va='center',color=color,fontsize=size,fontweight='bold' if abs(v['score']-best)<1e-9 and v['ok'] else 'normal')
        delta=stats((*k,'phasekv'),field)['score']-stats((*k,'full'),field)['score']
        ax.add_patch(Rectangle((len(ms)-.5,i-.5),1,1,facecolor=dmap(dn(delta)),edgecolor='white',lw=1.5))
        ax.text(len(ms),i,f'{delta:+.1f}',ha='center',va='center',color='white' if abs(delta)>32 else '#172033',fontsize=12,fontweight='bold')
    ax.set(xlim=(-.5,nc-.5),ylim=(len(keys)-.5,-.5),xticks=range(nc),xticklabels=[names[m] for m in ms]+['Δ P−Full'],yticks=range(len(keys)),yticklabels=labels)
    ax.tick_params(axis='both',length=0,pad=9);ax.xaxis.tick_top()
    for lab in ax.get_xticklabels():
        if lab.get_text() in ['PhaseKV','Δ P−Full']: lab.set_color('#b42335');lab.set_weight('bold')
    for s in ax.spines.values(): s.set_visible(False)
    for b in breaks: ax.axhline(b-.5,color='#94a3b8',lw=2)
    ax.axvline(len(ms)-.5,color='#64748b',lw=2)
    ax.set_title(title,loc='left',fontsize=17,fontweight='bold',pad=50,color='#142b4a')

labels=[]
for _,task,split,budget in lbkeys:
    metric='RL' if task in ['gov_report','qmsum','multi_news'] else 'Code' if task in ['lcc','repobench-p'] else 'F1'
    tag='复测' if split=='historical_retest' else '未见'
    n=stats(('longbench',task,split,budget,'full'))['n']
    labels.append(f'{task} · {metric}\n{tag} | {budget//1024}K | n={n}')
heat(fig.add_subplot(gs[:,0]),lbkeys,labels,methods,'A  LongBench｜10 个任务，按长度及历史划分',breaks=range(2,22,2))
rlabels=[f'{t}\n{b//1024}K | n=32' for _,t,_,b in rukeys]
heat(fig.add_subplot(gs[0,1]),rukeys,rlabels,methods,'B  RULER｜4 个任务 × 3 个长度',breaks=[3,6,9])
lkeys=lgkeys*3
llabels=[f'{k[1].upper()} · {label}' for label in ['官方得分','编号诊断','解析完成率'] for k in lgkeys]
fields=['score']*3+['indexed_accuracy_diagnostic']*3+['completion_pct']*3
heat(fig.add_subplot(gs[1,1]),lkeys,llabels,methods+extra,'C  LongGenBench｜每任务 16 条，上限 4096 输出',fields,[3,6])
fig.text(.04,.965,'PhaseKV 与各方法：本轮数据集质量详细对比',fontsize=28,fontweight='bold',color='#142b4a')
fig.text(.04,.936,'Llama-3.1-8B-Instruct · RTX 4090 D · FP16 · greedy · batch=1    |    质量生成 5456 条：成功 4751，OOM 705',fontsize=14,color='#475569')
fig.text(.04,.908,'阅读方式：同一行横向比较；蓝色越深得分越高。末列为 PhaseKV − Full（分数点），红色下降、蓝色上升。',fontsize=14,color='#334155')
cb1=fig.add_axes([.16,.075,.25,.014]); fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=qacmap),cax=cb1,orientation='horizontal',label='任务得分 / 完成率（0–100）')
cb2=fig.add_axes([.61,.075,.25,.014]); fig.colorbar(plt.cm.ScalarMappable(norm=dn,cmap=dmap),cax=cb2,orientation='horizontal',label='PhaseKV − Full（分数点）')
fig.text(.04,.035,'统计口径：主格为全部计划样本分母，OOM 记 0；灰格 OOM 为无成功结果，不是模型答错。部分成功格附成功数；未标注者全部成功。',fontsize=12,color='#475569')
fig.text(.04,.015,'F1＝QA F1，RL＝ROUGE-L，Code＝代码相似度；RULER 为官方字符串匹配，LongGenBench 为复合题准确率。不同指标不合成为“总准确率”。',fontsize=12,color='#475569')
name='09_quality_detailed'
fig.savefig(OUT/(name+'.png'),dpi=210,facecolor=fig.get_facecolor());fig.savefig(OUT/(name+'.svg'),facecolor=fig.get_facecolor());plt.close(fig)

def fmt(v): return '—' if v is None else f'{v:.2f}' if isinstance(v,float) else str(v)
def table(headers,data):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(x))+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table>'
agg=[]
for suite,task,split in dict.fromkeys(k[:3] for k in groups):
    vals=[]
    for m in methods:
        v=[r for r in rows if r['suite']==suite and r['task']==task and (r.get('evaluation_split') or 'new_frozen')==split and r['method']==m]
        vals.append(sum(r.get('score',0) for r in v)/len(v))
    agg.append([suite,task,split,*vals,vals[-1]-vals[0]])
coverage=[]
for suite in ['longbench','ruler','longgenbench']:
    coverage.append([suite,*[f"{sum(r['status']=='ok' for r in rows if r['suite']==suite and r['method']==m)}/{sum(r['suite']==suite and r['method']==m for r in rows)}" for m in methods]])
assert sum(r['suite']!='performance' and r['status']=='ok' for r in rows)==4751
assert sum(r['suite']!='performance' and r['status']=='oom' for r in rows)==705
notes=[
 'PhaseKV 在 LongBench 和 RULER 中的表现总体接近 Full，但 LongGenBench 长输出任务明显下降。Full、KIVI、SnapKV、PhaseKV 的质量生成全部完成，因此这四者的质量差异不受 OOM 零分影响。',
 'LongBench：跨两档长度按样本求均值，PhaseKV 的 qasper 为 53.07，Full 50.83、KIVI 50.67、SnapKV 52.35；multifieldqa_en 为 53.98，低于 KIVI 55.35 和 SnapKV 54.51。hotpotqa 和 2wikimqa 与 Full 接近。不能把观察到的微小差异解读为稳定优胜。',
 '摘要任务：PhaseKV 的 gov_report 为 31.94，Full 33.64、KIVI 34.06、SnapKV 30.48。相对 Full 的文档聚类配对差值为 -1.70，原有 95% bootstrap 区间 [-2.78,-0.61]，说明这是值得关注的下降。multi_news 为 25.00，对照 Full 25.68。qmsum 未见与复测样本分开显示。',
 '代码任务：PhaseKV 的 lcc 为 68.16，比 Full 70.31 低 2.16 分，但与 KIVI 67.91、SnapKV 68.05 接近；repobench-p 为 57.33，对照 Full 57.16，差别很小。代码相似度不能解释成答对题目的百分比。',
 'RULER：PhaseKV 的 multikey 和 vt 在三个长度上均为 100。multivalue 的 4K/8K/16K 为 100/97.66/98.44，Full 为 100/98.44/97.66。整体接近满分，存在天花板效应，不能证明更复杂推理也不退化。',
 'RULER 的 cwe 在 16K 才明显拉开：PhaseKV 72.19、KIVI 71.88、Full 70.94、SnapKV 43.44。PhaseKV 在本组比 SnapKV 高 28.75 分，但相对 Full 的任务级配对区间跨越 0，不能宣称统计显著优于 Full。',
 'LongGenBench：PhaseKV 的 GSM8K/MMLU/CSQA 官方分分别为 20.625/35.625/24.53125，Full 为 53.5417/48.125/75.9375；KIVI 为 46.6667/46.0417/74.6875。PhaseKV 相对 Full 下降 32.92/12.50/51.41 分。GSM8K 和 CSQA 在五个主方法中最低；MMLU 高于 SnapKV，但仍明显低于 Full 和 KIVI。',
 'LongGenBench 编号诊断：PhaseKV 的 GSM8K 为 30.00，Full 为 64.79；CSQA 为 24.53，Full 为 75.94。即使换成按 Answer 编号对齐的解析，退化仍存在。PhaseKV 解析完成率为 62.92%/73.54%/55.63%；CSQA 全部触及输出上限，已有重复 Answer_6 的原文证据。解析完成率不等同于正确率，低完成率也可能包含格式无法解析。',
 'SCOPE 的低系统分与 OOM 密切相关：LongBench 仅成功 200/640，RULER 128/384；RULER 的 8K 和 16K 全部 OOM，不能把图中的零系统分解释成模型答案全部错误。LongGenBench 中三个 SCOPE 版本的 MMLU 均成功 13/16，主版本成功子集得分 44.62，而失败记零后为 36.25；这一成功子集与其他方法的全体样本不是同一比较集合。',
 '历史划分：narrativeqa 全为复测；qmsum 未见只有 5 个样本索引，历史复测为 27 个。每个索引测两档长度。narrativeqa 和 qmsum 的历史样本包含共享上下文，聚类统计的独立上下文数均为 19。不能把重复索引、不同长度都视为独立文档。',
 '图中 Δ 为该行按样本求均值的差值；历史配对报告先在上下文簇内平均再跨簇平均，因此历史复测行的跨长度均值可能不同于配对报告。原有置信区间为 2000 次聚类 bootstrap，未做多重比较校正。本图格内加粗仅表示该行系统得分最高或并列最高，不代表显著性。'
]
detail=[]
for k in groups:
    v=stats(k);detail.append([*k,v['ok'],v['n'],v['success'],v['score']])
paired=json.loads((SRC/'quality_paired_vs_full.json').read_text(encoding='utf-8'))
ci=[[r['suite'],r['task'],r['evaluation_split'],r['paired_document_clusters'],r['delta_vs_full_points'],r['bootstrap95_low'],r['bootstrap95_high']] for r in paired if r['method']=='phasekv']
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>本轮质量对比详解</title><style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;max-width:1500px;margin:35px auto;padding:0 25px;color:#172033}h1,h2{color:#142b4a}img{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #dce2eb;text-align:right}th{background:#eef3fa}td:first-child{text-align:left}tr:nth-child(even){background:#f8fafc}a{color:#2563eb}</style><h1>本轮数据集质量对比详解</h1><p>所有分数均来自 native_llama_20261029；未修改原始结果，也未混入旧 Qwen 测试。</p><p><a href="09_quality_detailed.png">打开高清总图 PNG</a> · <a href="09_quality_detailed.svg">打开矢量图 SVG</a></p><a href="09_quality_detailed.png"><img src="09_quality_detailed.png"></a>'
doc+='<h2>逐项解读</h2>'+''.join('<p>'+html.escape(n)+'</p>' for n in notes)
doc+='<h2>质量生成完成情况</h2>'+table(['数据集',*[names[m] for m in methods]],coverage)
doc+='<h2>各任务均分（跨长度、失败记零；不同任务不再合并）</h2>'+table(['数据集','任务','划分',*[names[m] for m in methods],'PhaseKV−Full'],agg)
doc+='<h2>PhaseKV 与 Full 的原有配对置信区间</h2>'+table(['数据集','任务','划分','上下文簇数','差值','95%下界','95%上界'],ci)
doc+='<h2>全部细分数值：成功子集与系统分并列</h2>'+table(['数据集','任务','划分','预算','方法','成功','计划','成功子集均分','失败记零系统分'],detail)
doc+='<p>来源：records.jsonl、quality_paired_vs_full.json。生成脚本：plot_quality_detail.py。图中所有数值重新由逐条记录计算。</p></html>'
(OUT/'质量对比详解.html').write_text(doc,encoding='utf-8')
(OUT/'质量对比详解.txt').write_text('\n\n'.join(notes),encoding='utf-8')
report=OUT/'分析报告.html'
content=report.read_text(encoding='utf-8')
marker='<!-- detailed-quality-link -->'
if marker not in content:
    content=content.replace('<h2>主要发现与解释边界</h2>',marker+'<h2>新增：数据集质量详细总图</h2><p><a href="质量对比详解.html">打开质量对比详解（含高清总图、逐任务分析和失败分母）</a></p><a href="09_quality_detailed.png"><img src="09_quality_detailed.png" alt="数据集质量详细对比"></a><h2>主要发现与解释边界</h2>')
    report.write_text(content,encoding='utf-8')
print('Created: 09_quality_detailed.png / .svg and quality detail report; linked from original report.')
