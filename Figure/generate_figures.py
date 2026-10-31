"""Eight reproducible figures from audited reports; keep cohorts separate."""
from pathlib import Path
import hashlib, html, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'axes.unicode_minus':False,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BDC7D2',
 'text.color':'#253449','axes.labelcolor':'#253449','xtick.color':'#536378',
 'ytick.color':'#536378','figure.facecolor':'white','savefig.facecolor':'white',
 'pdf.fonttype':42,'svg.fonttype':'path','legend.frameon':False})
C={'full':'#8896A6','kivi':'#5479B6','snapkv':'#8B6DB0','scope':'#D2A447','phasekv':'#168C8B','p5':'#E47143'}
LABELS={'full':'Full','kivi':'KIVI','snapkv':'SnapKV','scope':'SCOPE','phasekv':'PhaseKV'}
METHODS=list(LABELS)
paths={'algorithms':'reports/4090d_comparison_20261025/comparison.json',
 'sweep':'reports/phasekv_sweep_20261026/analysis.json',
 'retest':'reports/phasekv_prompt_retest_20261027/analysis.json',
 'optimized':'reports/phasekv_long_output_20261027/analysis.json'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
src={k:read(ROOT/v) for k,v in paths.items()}
alg,sweep,retest,opt=(src[k] for k in paths)
assert alg['audit']['input_hash_mismatches']==0 and alg['audit']['checkpoint_and_data_equal']
for item in alg['audit']['source_files']:
    assert hashlib.sha256(Path(item['file']).read_bytes()).hexdigest()==item['sha256']
for stage,digest in opt['record_hashes'].items():
    assert hashlib.sha256((ROOT/opt['source']/stage/'records.jsonl').read_bytes()).hexdigest()==digest
assert opt['audit']=='passed' and retest['statuses']['ok']==8864
FIGURES=[];DATA={}
def save(fig,stem,title,note,data):
    for fmt in ('png','pdf','svg'):fig.savefig(OUT/(stem+'.'+fmt),dpi=300,bbox_inches='tight',pad_inches=.16)
    FIGURES.append(dict(stem=stem,title=title,notes=note));DATA[stem]=data
    plt.close(fig);print(stem,flush=True)
def canvas(title,subtitle,cols=2,rows=1,size=(14,7),bottom=.25):
    fig,axs=plt.subplots(rows,cols,figsize=size,squeeze=False)
    fig.subplots_adjust(left=.08,right=.96,top=.78,bottom=bottom,wspace=.30,hspace=.50)
    fig.text(.06,.95,title,fontsize=20,fontweight='bold',va='top')
    fig.text(.06,.875,subtitle,fontsize=11,color='#607086',va='top')
    return fig,axs.ravel()
def foot(fig,text):fig.text(.06,.045,text,fontsize=9,color='#607086',va='bottom',linespacing=1.65)
def grid(ax,axis='y'):ax.grid(axis=axis,color='#E5EBF1',lw=.8);ax.set_axisbelow(True)
def ret(r,b,base='Full-matched'):return retest['rows'][f'P{r}-B{b}-W128-C4-D1'][base]

def fig1():
    fig,axs=canvas('01  Prompt 保留比例与量化位宽','受控复测：70% / 100% × 2 / 4 / 8-bit；固定热区128、冷区4-bit、生成保留100%')
    x=np.arange(3);data={}
    for ax,key,title in zip(axs,('qa','summary'),('问答质量 · F1 ↑','摘要质量 · ROUGE-L ↑')):
        for r,color,marker,offset in ((.7,C['phasekv'],'o',-18),(1,C['kivi'],'s',12)):
            vals=[ret(r,b)[key] for b in (2,4,8)]
            ax.plot(x,vals,marker=marker,lw=2.3,ms=7,color=color,label=f'{int(r*100)}% 保留')
            label_offset=(12 if r==.7 else -18) if key=='qa' else offset
            for xx,y in zip(x,vals):ax.annotate(f'{y:.2f}',(xx,y),xytext=(0,label_offset),textcoords='offset points',ha='center',fontsize=10,color=color)
            data[f'{r}_{key}']=vals
        full=retest['rows']['Full-matched']['Full-matched'][key]
        ax.axhline(full,ls='--',lw=1.5,color=C['full'],label=f'Full-matched ({full:.2f})')
        ax.set_xticks(x,['2-bit','4-bit','8-bit']);ax.set_xlim(-.25,2.25)
        ax.set_ylim((35.3,38.1) if key=='qa' else (25.9,27.3));ax.set_title(title,loc='left',pad=15);grid(ax)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),fontsize=9,ncol=2)
    note='来源：2026-10-27 prompt复测；同一批256篇文档 × 4种输入预算，按任务等权。\n折线纵轴局部放大；均值差异不等于统计显著。Full-matched采用一致prefill路径。'
    foot(fig,note);save(fig,'01_prompt_retention_and_bits','Prompt保留率与位宽消融',note,data)

def fig2():
    fig,axs=canvas('02  热区、冷区位宽与生成保留率','探索性筛选：摘要 ROUGE-L；Prompt固定2-bit，分开呈现70%与100%保留',2,2,(14,9.5),.18)
    fig.subplots_adjust(right=.86,left=.10,hspace=.55)
    entries={r['name']:r for r in sweep['stages']['decode'] if r['spec']['method']=='phasekv'};data={}
    for col,r in enumerate((.7,1)):
        for row in (0,1):
            ax=axs[row*2+col];windows=[32,64,128] if row==0 else [64,128];xs=[2,4,8] if row==0 else [.5,.75,1]
            vals=np.array([[entries[f'P{r}-B2-W{w}-C{x if row==0 else 4}-D{1 if row==0 else x}']['summary'] for x in xs] for w in windows])
            im=ax.imshow(vals,cmap='YlGnBu',vmin=28.5,vmax=30.1,aspect='auto')
            for iy in range(len(windows)):
                for ix in range(3):ax.text(ix,iy,f'{vals[iy,ix]:.2f}',ha='center',va='center',fontsize=12,color='white' if vals[iy,ix]>29.4 else '#17354B')
            ax.set_xticks(range(3),[f'{x}-bit' for x in xs] if row==0 else [f'{int(x*100)}%' for x in xs]);ax.set_yticks(range(len(windows)),[f'W={w}' for w in windows])
            ax.set_xlabel('冷区位宽（生成保留100%）' if row==0 else '生成保留比例（冷区4-bit）');ax.set_title(f'Prompt保留{int(r*100)}%',loc='left',pad=12)
            data[f'{r}_{row}']=dict(windows=windows,x=xs,summary=vals.tolist())
    cax=fig.add_axes([.90,.24,.015,.49]);fig.colorbar(im,cax=cax,label='摘要 ROUGE-L ↑')
    note='来源：2026-10-26 decode筛选；每配置4任务 × 8篇 × 2种预算，其中摘要2任务。\n小样本探索，不与正式复核分数混算；QA输出普遍较短，不能据其不变推断冷区参数没有影响。'
    foot(fig,note);save(fig,'02_decode_parameter_screening','热区与冷区参数筛选',note,data)

def fig3():
    fig,axs=canvas('03  六组配置相对 Full-matched 的质量变化','点为配对均值差；线为按文档聚类的95% bootstrap区间；负值表示下降',size=(14,7.2),bottom=.23)
    names=[(r,b) for r in (.7,1) for b in (2,4,8)];data={}
    for ax,key,title in zip(axs,('qa','summary'),('问答 F1 差值','摘要 ROUGE-L 差值')):
        ax.axvline(0,color='#6A7789',lw=1.3);ax.axvline(-1,color='#C49567',ls='--',lw=1,label='−1分参考线')
        for i,(r,b) in enumerate(names):
            m=ret(r,b);delta=-m[key+'_loss'];lo,hi=m[key+'_delta_ci95']
            ax.errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color=C['phasekv'] if r==.7 else C['kivi'],capsize=4,ms=7,lw=1.7)
            data[f'{r}_{b}_{key}']=dict(delta=delta,ci95=[lo,hi])
        ax.set_yticks(range(6),[f'{int(r*100)}% / {b}-bit' for r,b in names]);ax.invert_yaxis()
        ax.set_xlabel('PhaseKV − Full-matched（分）');ax.set_title(title,loc='left',pad=15);grid(ax,'x');ax.legend(loc='lower left',fontsize=9)
    note='来源：2026-10-27 prompt复测；同一文档的4种预算一起重采样，未校正多重比较。\n区间跨0时不能断言提升/下降；均值损失小于1分，不等于95%置信度下损失也小于1分。'
    foot(fig,note);save(fig,'03_quality_deltas_with_uncertainty','六组配置的配对质量区间',note,data)

def fig4():
    fig,axs=canvas('04  70% / 4-bit：速度收益随工作负载变化','参数复测中的 PhaseKV；同轮 Full-native 对照；尚未应用本次 P1—P5 优化',size=(14.5,7.6),bottom=.23)
    rows=ret(.7,4,'Full-native')['shapes'];y=np.arange(len(rows));labels=[f"{r['input']:,} + {r['output']:,}" for r in rows]
    ax=axs[0];vals=[r['speed_vs_full'] for r in rows]
    bars=ax.barh(y,vals,color=[C['phasekv'] if v>=1 else '#D29970' for v in vals],height=.6)
    ax.axvline(1,color=C['full'],ls='--');ax.bar_label(bars,labels=[f'{v:.2f}×' for v in vals],padding=5,fontsize=10)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,2);ax.set_xlabel('解码吞吐 / Full-native');ax.set_title('速度：超过1表示更快',loc='left');grid(ax,'x')
    full={(r['input'],r['output']):r for r in retest['rows']['Full-native']['Full-native']['shapes']}
    ax=axs[1]
    ax.barh(y-.17,[full[(r['input'],r['output'])]['kv_mib'] for r in rows],height=.30,color=C['full'],label='Full-native')
    ax.barh(y+.17,[r['kv_mib'] for r in rows],height=.30,color=C['phasekv'],label='PhaseKV 70% / 4-bit')
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('最终实际 KV 容量（MiB）↓');ax.set_title('存储：固定输入与输出长度',loc='left');grid(ax,'x');ax.legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
    note='来源：2026-10-27 prompt复测；每形状3次正式测量，速度与KV取中位数，预热不计入。\n左侧工作负载标签为“输入 + 输出 token”；decode不含prefill，但包含缓存维护。'
    foot(fig,note);save(fig,'04_phasekv_speed_and_kv_by_workload','70%/4-bit的分场景速度与KV',note,rows)

ALG_NOTE='历史五算法批次：同模型、同数据/输入，PhaseKV单独复跑；各方法prefill路径不完全相同。'
PARAM_NOTE='PhaseKV=P30/B2/W32/C4/D1；KIVI=2-bit/R64；SnapKV=P30%；SCOPE=ALLKV+Slide/生成预算32。'

def fig5():
    fig,axs=canvas('05  五算法质量对比','历史同数据实验；展示实际推理流程的质量，不将全部差值归因于压缩算法',bottom=.26)
    rows={r['method']:r for r in alg['quality']};data={}
    for ax,key,title in zip(axs,('qa_f1','rouge_l'),('问答 F1 ↑','摘要 ROUGE-L ↑')):
        vals=[rows[m][key] for m in METHODS]
        bars=ax.bar(range(5),vals,color=[C[m] for m in METHODS],width=.65);ax.bar_label(bars,fmt='%.2f',padding=5,fontsize=10)
        ax.set_xticks(range(5),[LABELS[m] for m in METHODS]);ax.set_ylim(0,42 if key=='qa_f1' else 31)
        ax.set_title(title,loc='left',pad=15);grid(ax);data[key]=vals
    note=ALG_NOTE+'\n'+PARAM_NOTE+'\n来源：2026-10-25对比报告；每方法1024条质量记录，QA与摘要分别按任务等权；不含P5质量数据。'
    foot(fig,note);save(fig,'05_algorithm_quality_comparison','五算法质量对比',note,data)

def fig_perf(stem,sweep_name,title,xkey,selected):
    fig,axs=canvas(title,'历史批次 · PhaseKV：30%保留 / Prompt 2-bit / 热区32 · Qwen2.5-7B-Instruct · RTX 4090 D',size=(14.4,7.2),bottom=.27)
    data={}
    for method in METHODS:
        rr=sorted([r for r in alg['performance'][method] if r['sweep']==sweep_name and r[xkey] in selected],key=lambda r:r[xkey])
        assert len(rr)==len(selected),(method,sweep_name)
        assert all(r['n']==3 for r in rr)
        xs=np.array([r[xkey] for r in rr]);ys=[r['decode_tps'] for r in rr]
        error=np.array([[r['decode_tps']-r['decode_tps_min'] for r in rr],[r['decode_tps_max']-r['decode_tps'] for r in rr]])
        axs[0].errorbar(xs,ys,yerr=error,lw=2,marker='o',ms=5,color=C[method],label=('PhaseKV (30% / 2-bit / W32)' if method=='phasekv' else LABELS[method]),capsize=2)
        axs[1].plot(xs,[r['final_kv_mib'] for r in rr],lw=2,marker='o',ms=5,color=C[method],label=('PhaseKV (30% / 2-bit / W32)' if method=='phasekv' else LABELS[method]))
        data[method]=rr
    for ax in axs:
        ax.set_xticks(selected,[f'{v:,}' for v in selected]);grid(ax)
        ax.set_xlabel('输入 token（输出固定256）' if xkey=='input_tokens' else '输出 token（输入固定1024）')
        ax.set_ylim(bottom=0)
    axs[0].set_ylabel('解码吞吐（token/s）↑');axs[0].set_title('解码速度',loc='left',pad=15)
    axs[1].set_ylabel('最终实际 KV（MiB）↓');axs[1].set_title('实际缓存容量',loc='left',pad=15)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.155),ncol=5,fontsize=10)
    note=ALG_NOTE+'\n'+PARAM_NOTE+'\n每点为3次正式测量的中位数；速度误差棒为实测最小—最大值，非置信区间。'
    foot(fig,note);save(fig,stem,title,note,data)

def fig8():
    fig,axs=canvas('08  优化前后 PhaseKV，以及其他压缩方法','1024输入 + 4096输出 · 本图PhaseKV：70%保留 / Prompt 4-bit / 热区128；不同于图07的30% / 2-bit / 热区32',3,1,(18.5,8.2),.27)
    row=opt['metrics'][0];a,z=(row['stages'][k] for k in ('baseline','p5'))
    ax=axs[0];bars=ax.bar([0,1],[a['decode_tokens_per_s'],z['decode_tokens_per_s']],color=[C['phasekv'],C['p5']],width=.6)
    ax.bar_label(bars,fmt='%.2f',padding=7,fontsize=13,fontweight='bold');ax.set_xticks([0,1],['PhaseKV 70%/4-bit\n优化前 · W128','PhaseKV 70%/4-bit\nP5优化后 · W128'])
    ax.set_ylim(0,49);ax.set_ylabel('解码吞吐（token/s）↑');ax.set_title('A  同轮实测：各1次正式测速',loc='left',pad=16)
    ax.text(.5,44,f"+{row['throughput_change_percent']:.2f}%",ha='center',fontsize=21,fontweight='bold',color=C['p5']);grid(ax)
    historic=[]
    for m in METHODS[:-1]:
        rr=[r for r in alg['performance'][m] if r['sweep']=='output_sweep' and r['input_tokens']==1024 and r['output_tokens']==4096]
        assert len(rr)==1;historic.append(rr[0])
    labels=['Full','KIVI','SnapKV','SCOPE','PhaseKV 70%/4-bit\n优化前 · W128','PhaseKV 70%/4-bit\nP5优化后 · W128']
    colors=[C[m] for m in METHODS[:-1]]+[C['phasekv'],C['p5']]
    values=[[r['decode_tps'] for r in historic]+[a['decode_tokens_per_s'],z['decode_tokens_per_s']],
            [r['final_kv_mib'] for r in historic]+[a['kv_mib'],z['kv_mib']]]
    for ax,vals,title,xlabel in zip(axs[1:],values,('B  吞吐：跨批次参考','C  KV容量：跨批次参考'),('解码吞吐（token/s）↑','最终实际 KV（MiB）↓')):
        bars=ax.barh(np.arange(6),vals,color=colors,height=.65)
        for bar in bars[:4]:bar.set_hatch('///');bar.set_edgecolor('white')
        ax.bar_label(bars,fmt='%.2f',padding=5,fontsize=10);ax.axhline(3.5,color='#ABB7C5',ls='--',lw=1)
        ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.24)
        ax.set_xlabel(xlabel);ax.set_title(title,loc='left',pad=16);grid(ax,'x')
    handles=[Patch(facecolor='#A6B2C1',edgecolor='white',hatch='///',label='历史五算法批次：3次正式测量中位数'),Patch(facecolor=C['p5'],label='最新优化A/B批次：各1次正式测量')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.165),ncol=2,fontsize=10)
    note='本图33.49→37.05为同配置A/B；图07的40.66来自不同配置，不能连成优化前后变化。P5未开启独立GQA复用。P5实际KV增加2.71 MiB，来自空闲页容量。\n历史参考：KIVI 2-bit/R64、SnapKV保留30%、SCOPE ALLKV+Slide/预算32；未做等KV预算或等质量配平。\n右两图只能作跨批次参考，不能据此确认P5优于其他算法。P5未测准确率；两版输出从第92个token开始分叉。'
    foot(fig,note);save(fig,'08_optimized_phasekv_and_other_algorithms','优化前后PhaseKV及其他算法',note,dict(same_run=row,historical_methods=dict(zip(METHODS[:-1],historic)),cross_run_reference_only=True))

def main():
    fig1();fig2();fig3();fig4();fig5()
    fig_perf('06_algorithm_long_context','context_sweep','06  长输入：五算法速度与 KV 占用','input_tokens',[1024,4096,8192,16384,24576,32000])
    fig_perf('07_algorithm_long_output','output_sweep','07  长输出：五算法速度与 KV 占用','output_tokens',[256,1024,2048,3072,4096])
    fig8()
    metadata={'sources':{k:dict(path=v,sha256=hashlib.sha256((ROOT/v).read_bytes()).hexdigest()) for k,v in paths.items()},'figures':FIGURES,'data':DATA,'matplotlib':matplotlib.__version__}
    (OUT/'plot_data.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    guide=['PhaseKV 实验图集','================','8张图，每张PNG（300 dpi）、PDF和SVG矢量版本。',
           'PDF嵌入字体，SVG文字转路径；绘图数值与来源哈希见plot_data.json。','']
    guide += [f"{f['stem']}：{f['title']}\n{f['notes']}\n" for f in FIGURES]
    guide += ['没有纳入PyramidKV。历史PhaseKV参数与70%/4-bit复测不同，已在各图标注。',
              '第08图只有优化前/后是最新同轮实验；其他算法为历史参考，不是同轮排名。',
              '没有P5质量数据，未补画准确率；没有逐项P1—P5 GPU结果，未虚构加速消融。',
              '复现：项目根目录运行 .venv/Scripts/python.exe Figure/generate_figures.py。',
              '依赖Matplotlib、NumPy；Windows中文字体使用Microsoft YaHei。']
    (OUT/'README.txt').write_text('\n'.join(guide)+'\n',encoding='utf-8')
    cards=''.join(f'<section><h2>{html.escape(f["title"])}</h2><img loading="lazy" src="{f["stem"]}.png"><p>{html.escape(f["notes"]).replace(chr(10),"<br>")}</p><p><a href="{f["stem"]}.png">PNG</a> · <a href="{f["stem"]}.pdf">PDF</a> · <a href="{f["stem"]}.svg">SVG</a></p></section>' for f in FIGURES)
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhaseKV实验图集</title><style>body{font:16px system-ui;line-height:1.7;color:#253449;background:#f4f7fb;max-width:1450px;margin:30px auto;padding:20px}section{background:white;padding:25px;margin:25px 0;border-radius:12px}img{width:100%;height:auto}p{color:#607086}a{color:#168C8B}</style><h1>PhaseKV 实验图集</h1><p>8张图 · 参数消融 / 多算法比较 / P5优化前后 · 2026-10-27</p>'+cards+'</html>',encoding='utf-8')

if __name__=='__main__':main()
