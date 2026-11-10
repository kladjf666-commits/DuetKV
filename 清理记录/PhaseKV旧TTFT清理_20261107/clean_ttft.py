"""Remove superseded PhaseKV TTFT fields only; retain an audit without old values."""
from pathlib import Path
import collections
import hashlib
import json
import re
import os

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'A800最终测试'
AUDIT = Path(__file__).resolve().parent
REPORT = AUDIT / '清理与校验.json'
assert not REPORT.exists(), 'Already completed; do not repeat.'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def relative(p):
    return p.relative_to(ROOT).as_posix()

def safe(p):
    assert p.resolve().is_relative_to(ROOT)
    assert not any(x.is_symlink() or (hasattr(x, 'is_junction') and x.is_junction()) for x in [p, *p.parents] if x.is_relative_to(ROOT))

def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def dump(d):
    return (json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')

def family(s):
    return str(s).lower().startswith('phasekv') or 'bettermemory' in str(s).lower()

changed = []
pending = {}
removed = {}

def scrub(o, paths, path=''):
    if isinstance(o, dict):
        result = {}
        for k, v in o.items():
            if 'ttft' in k.lower():
                paths.append(path + '/' + k)
            else:
                result[k] = scrub(v, paths, path + '/' + k)
        return result
    if isinstance(o, list):
        return [scrub(v, paths, path + '/' + str(i)) for i, v in enumerate(o)]
    return o

def mixed_speed(d, paths):
    for method in d.get('speed', {}):
        if family(method):
            d['speed'][method] = scrub(d['speed'][method], paths, '/speed/' + method)

# Snapshot every existing result file, both code trees, and the retest delivery.
snapshot_roots = [BASE, ROOT/'Code-Original-NoChanging - A800', ROOT/'Code-Original-NoChanging - A800 - BetterMemory', ROOT/'A800复现资料/TTFT同卡复测']
before = {}
for directory in snapshot_roots:
    for p in directory.rglob('*'):
        if p.is_file():
            safe(p)
            before[relative(p)] = sha(p)
(AUDIT/'操作前指纹.json').write_bytes(dump(before))

old_dirs = ['多算法最终对照', '优化前_Prompt90', 'BetterMemory_速度与峰值显存']
for sub in old_dirs:
    for p in (BASE/sub).rglob('*.json'):
        old = load(p)
        d = json.loads(json.dumps(old))
        paths = []
        rp = p.relative_to(BASE).as_posix()
        if rp == '优化前_Prompt90/分析与图表/analysis.json':
            mixed_speed(d, paths)
            for key in ['gates', 'speed_changes']:
                d[key] = scrub(d[key], paths, '/' + key)
        elif rp == '优化前_Prompt90/分析与图表/速度详细图表/绘图数据.json':
            cells = []
            for i, c in enumerate(d['cells']):
                if c.get('metric') == 'ttft' and family(c.get('method')):
                    paths.append('/cells/' + str(i))
                else:
                    cells.append(c)
            d['cells'] = cells
            d['relative_changes'] = scrub(d['relative_changes'], paths, '/relative_changes')
        elif rp == 'BetterMemory_速度与峰值显存/全部算法速度与峰值对比.json':
            mixed_speed(d, paths)
        elif rp == '多算法最终对照/分析与图表/analysis.json':
            points = d['speed_memory_analysis']['points']
            for i, point in enumerate(points):
                if family(point['method']):
                    points[i] = scrub(point, paths, '/speed_memory_analysis/points/' + str(i))
        elif rp.startswith(('多算法最终对照/phasekv/', '优化前_Prompt90/', 'BetterMemory_速度与峰值显存/')):
            d = scrub(d, paths)
        if d != old:
            # Check every difference is exactly one of the explicitly selected deletions.
            check = json.loads(json.dumps(old))
            # Delete list records in reverse index order to keep pointers stable.
            for pointer in sorted(paths, key=lambda s: (s.count('/'), int(s.rsplit('/',1)[-1]) if s.rsplit('/',1)[-1].isdigit() else -1), reverse=True):
                keys = pointer.strip('/').split('/')
                target = check
                for k in keys[:-1]:
                    target = target[int(k)] if isinstance(target, list) else target[k]
                if isinstance(target, list): del target[int(keys[-1])]
                else: del target[keys[-1]]
            assert check == d, relative(p)
            pending[p] = dump(d)
            removed[relative(p)] = paths

p = BASE/'TTFT同卡复测_20261107/analysis.json'
d = load(p)
paths = []
for case, point in d['points'].items():
    assert 'historical_bettermemory_ttft_s' in point
    del point['historical_bettermemory_ttft_s']
    paths.append('/points/' + case + '/historical_bettermemory_ttft_s')
pending[p] = dump(d)
removed[relative(p)] = paths

def text_of(p):
    return pending.get(p, p.read_bytes()).decode('utf-8-sig')

def edit(path, replacements=(), transform=None):
    p = ROOT/path
    s = text_of(p)
    for old, new in replacements:
        assert old in s, (path, old)
        s = s.replace(old, new)
    if transform: s = transform(s)
    pending[p] = s.encode('utf-8')

NOTICE = '按用户要求，旧PhaseKV、Prompt90及BetterMemory的TTFT字段与相关比较已清理；其它算法TTFT及所有质量、解码/端到端速度、显存指标保留。PhaseKV的TTFT仅采用TTFT同卡复测_20261107中的两个测点，不补齐或拼接成16点成绩。'

def bm_report(s):
    lines = []
    for line in s.splitlines():
        if ' | ' in line:
            cells = line.split(' | ')
            assert len(cells) == 6
            del cells[4]
            line = ' | '.join(cells)
        line = re.sub(r'；TTFT[^；\n]*秒（[^）]*）', '', line)
        lines.append(line)
    return '\n'.join(lines) + '\n\n' + NOTICE + '\n'

edit('A800最终测试/BetterMemory_速度与峰值显存/结果查验与分析.txt', [
    ('速度/TTFT取3轮中位数', '速度取3轮中位数'),
    ('；TTFT全部增加，端到端全部下降。', '；端到端全部下降。'),
], bm_report)
edit('A800最终测试/BetterMemory_速度与峰值显存/阅读说明.txt', [('速度、TTFT及prefill', '速度及prefill')], lambda s:s+'\n'+NOTICE+'\n图表/05_首token等待对比仅保留其它算法的TTFT。\n')

for name in ['速度与峰值汇总.json', '实验身份.json']:
    p = BASE/'BetterMemory_速度与峰值显存'/name
    d = json.loads(text_of(p))
    d['note'] = d['note'].replace('速度/TTFT', '速度').replace('速度、TTFT、', '速度、') + ' 旧TTFT已清理；PhaseKV TTFT以同卡复测两点为准。'
    pending[p] = dump(d)

def p90_report(s):
    lines=[]
    percent_table=False
    performance_table=False
    for line in s.splitlines():
        if line.startswith('32768+2048/B8 | ') and 'TTFT秒' in line:
            performance_table=True
        elif not line.strip():
            performance_table=False
        if line.startswith('shape_B | ') and ' | TTFT | ' in line: percent_table=True
        if percent_table and ' | ' in line:
            cells=line.split(' | ')
            assert len(cells)==6
            del cells[3]
            line=' | '.join(cells)
        elif percent_table: percent_table=False
        if performance_table and line.startswith(('旧PhaseKV | ', '新PhaseKV90% | ')):
            cells=line.split(' | ')
            assert len(cells)==8
            cells[3]='已清理'
            line=' | '.join(cells)
        line=re.sub(r'，但TTFT([\d.]+)秒明显短于[\d.]+秒，', r'；KIVI的TTFT为\1秒，', line)
        if line.startswith('TTFT也更长，'):line='PhaseKV旧TTFT已清理，首token延迟请查看最新同卡复测。'
        if line.startswith('原始结果保持不变：'):line='记录位于A800最终测试/优化前_Prompt90/；旧TTFT已清理，其余原记录字段保留。'
        lines.append(line)
    return '\n'.join(lines)+'\n\n'+NOTICE+'\n'

edit('A800最终测试/优化前_Prompt90/分析与图表/结果查验与分析.txt', transform=p90_report)
edit('A800最终测试/优化前_Prompt90/分析与图表/速度详细图表/阅读说明.txt', [
    ('04 首token延迟：预填充、选择/量化与首token，单位秒，越低越好。', '04 首token延迟：仅保留其它算法，单位秒，越低越好；PhaseKV旧TTFT已移除。'),
    ('05 新旧PhaseKV逐点变化：相对旧组的百分比，绿色改善、红色退步。', '05 新旧PhaseKV逐点变化：仅保留解码与端到端吞吐，相对旧组的百分比，绿色改善、红色退步。'),
    ('前四图：每列为相同输入长度/输出长度/Batch，颜色仅在该列归一化。', '01–03图：每列为相同输入长度/输出长度/Batch，颜色仅在该列归一化。04图重绘后使用统一秒数色标，深色延迟更低。'),
    ('深色表示该列更好，不可跨列仅用颜色判断绝对快慢；绝对值直接标在格子中。', '01–03图深色表示该列更好，不可跨列仅用颜色判断绝对快慢；绝对值直接标在格子中。'),
    ('金框是该列最佳，蓝框是新PhaseKV；OOM与未测分别列出，绝不填零。', '01–03图金框是该列最佳，蓝框是新PhaseKV；各图OOM与未测分别列出，绝不填零。'),
    ('没有重新执行GPU测试或改变任何原始结果。绘图数据.json保存数值与源文件指纹。', '没有重新执行GPU测试。旧PhaseKV TTFT字段与图表项已清理，其余指标保留。绘图数据.json的source_sha256是清理前来源指纹；当前文件指纹见清理记录。'),
])
edit('A800最终测试/多算法最终对照/分析与图表/最终结果报告.txt', transform=lambda s:s+'\n'+NOTICE+'\n')

def latest_report(s):
    s=re.sub(r'^此前同一optimized代码：.*$', '旧PhaseKV/BetterMemory的TTFT数值已按用户要求清理，本次同卡配对记录完整保留。', s, flags=re.M)
    s=re.sub(r'此前[\d.%–—-]+的跨运行涨幅', '历史跨运行观测', s)
    s=s.replace('旧结果额外约20秒', '旧结果计时差异')
    s=s.replace('原16点结果继续独立保存，本次只新增两个测点，不将本次较快的TTFT拼接成一轮新的16点结果。', '原16点的质量、速度与显存继续独立保存，旧PhaseKV TTFT已清理；本次仅保留两个测点，不拼接成16点TTFT结果。')
    return s

edit('A800最终测试/TTFT同卡复测_20261107/结果查验与分析.txt', transform=latest_report)
edit('README.md', [('BetterMemory本轮保留速度、TTFT和峰值显存', 'BetterMemory原16点保留速度和峰值显存')], lambda s:s+'\n'+NOTICE+'\n最新记录：[TTFT同卡复测分析](A800最终测试/TTFT同卡复测_20261107/结果查验与分析.txt)。\n')
edit('项目文件结构.md', [('48条正式测量、16条预热、TTFT、各阶段峰值', '48条正式测量、16条预热、各阶段峰值')], lambda s:s+'\n'+NOTICE+'\n此次字段清理、图表调整及完整性校验见 `清理记录/PhaseKV旧TTFT清理_20261107/`；旧来源清单仍是历史快照。\n')
edit('A800_HANDOFF.md', [
    ('此前16点BetterMemory记录的TTFT普遍高于旧P90，且设备不同；最新同卡配对复测未复现大幅退化。旧记录仍原样保留，不以新两点替换或拼接原16点成绩。', NOTICE),
    ('原核心算法和既有最终结果不变。', '核心算法未改动；历史结果仅清理PhaseKV系列旧TTFT，质量、速度及显存保持原值。'),
])
edit('A800最终测试/阅读说明.txt', [
    ('48正式轮速度/TTFT/prefill', '48正式轮速度/prefill'),
    ('本轮BetterMemory速度、TTFT与峰值保留', '原16点BetterMemory速度与峰值保留'),
], lambda s:s+'\n'+NOTICE+'\n')
edit('A800最终测试/README.md', [
    ('BetterMemory 本轮速度、首 token 延迟和峰值显存', 'BetterMemory 原16点速度和峰值显存'),
    ('解码、端到端、首token延迟图', '解码、端到端及其它算法首token延迟图'),
    ('单请求吞吐、解码耗时、TTFT、prefill', '单请求吞吐、解码耗时、prefill'),
    ('`05_首token等待对比`，每图', '`05_首token等待对比`（仅其它算法），每图'),
    ('总/单请求/端到端吞吐、TTFT、各阶段峰值', '总/单请求/端到端吞吐、各阶段峰值'),
    ('速度和 TTFT 取3轮正式测量的中位数', '速度取3轮正式测量的中位数'),
    ('端到端吞吐、TTFT、新旧 PhaseKV 变化', '端到端吞吐、其它算法TTFT、新旧 PhaseKV 吞吐变化'),
], lambda s:s+'\n## 8. 旧TTFT清理说明\n\n'+NOTICE+'\n\n历史PhaseKV系列JSON现为移除TTFT后的记录，目录名称中的“原始记录”沿用原名。最新复测results/、test_package/及回传压缩包保持原字节；其它算法独立结果目录也未修改。旧来源哈希表示清理前版本，本次文件指纹和校验见 `清理记录/PhaseKV旧TTFT清理_20261107/`。\n')

# All JSON transformations have been checked before any result is modified.
for p, content in pending.items():
    safe(p)
    changed.append({'path':relative(p),'before_sha256':sha(p),'after_sha256':hashlib.sha256(content).hexdigest(),'removed_paths':removed.get(relative(p),[])})
    p.write_bytes(content)

# Redraw mixed charts, retaining other methods' values and non-TTFT PhaseKV panels.
os.environ['MPLCONFIGDIR'] = str(AUDIT/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'svg.fonttype':'none'})
p90 = load(BASE/'优化前_Prompt90/分析与图表/analysis.json')
methods = [m for m in p90['speed'] if not family(m)]
cases = list(p90['speed']['full'])
names={'full':'Full','kivi':'KIVI','clusterkv':'ClusterKV','snapkv':'SnapKV','pyramidkv':'PyramidKV','million':'MILLION','scope':'SCOPE'}
arr = np.full((len(methods),len(cases)),np.nan)
for i,m in enumerate(methods):
    for j,c in enumerate(cases):
        value=p90['speed'][m].get(c,{}).get('ttft')
        if value is not None: arr[i,j]=value
fig,ax=plt.subplots(figsize=(19,6.2))
cm=plt.get_cmap('Blues_r').copy();cm.set_bad('#eeeeee')
im=ax.imshow(arr,cmap=cm,aspect='auto',vmin=0,vmax=float(np.nanmax(arr)))
ax.set_xticks(range(len(cases)),[c.replace('_B','\nB') for c in cases],fontsize=9)
ax.set_yticks(range(len(methods)),[names[m] for m in methods],fontsize=11)
for i,m in enumerate(methods):
    for j,c in enumerate(cases):
        rec=p90['speed'][m].get(c,{})
        if np.isfinite(arr[i,j]):
            label=f'{arr[i,j]:.2f}'
            color='white' if arr[i,j]<np.nanmax(arr)*.45 else '#172033'
        else:
            status=str(rec.get('status','not_tested'))
            label='OOM' if 'oom' in status.lower() else '未测'
            color='#666666'
        ax.text(j,i,label,ha='center',va='center',fontsize=9,color=color)
ax.set_title('其它算法首token延迟：16个场景（秒，越低越好）',fontsize=17,pad=18)
fig.colorbar(im,ax=ax,pad=.015,label='TTFT（秒）')
fig.text(.05,.025,'旧PhaseKV / Prompt90 / BetterMemory的TTFT已清理；最新两点同卡复测单独保存。其它算法数值保持原值。',fontsize=10)
fig.subplots_adjust(left=.08,right=.94,top=.87,bottom=.18)

chart_paths=[]
def save_chart(fig, stem):
    for ext in ['png','svg']:
        p=BASE/(stem+'.'+ext);safe(p)
        old_hash=sha(p)
        fig.savefig(p,dpi=150,metadata={'Date':None} if ext=='svg' else None)
        chart_paths.append(relative(p))
        changed.append({'path':relative(p),'before_sha256':old_hash,'after_sha256':sha(p),'change':'Rebuilt without historical PhaseKV TTFT; other series retained.'})

save_chart(fig,'优化前_Prompt90/分析与图表/速度详细图表/04_首token延迟_全部16点')
save_chart(fig,'BetterMemory_速度与峰值显存/图表/05_首token等待对比')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,9),sharey=True)
for ax,metric,title in zip(axes,['tps','e2e_tps'],['解码总吞吐变化','端到端吞吐变化']):
    values=[p90['speed_changes'][c]['phasekv'][metric] for c in cases]
    ax.barh(range(len(cases)),values,color=['#268565' if v>=0 else '#ca554b' for v in values])
    for i,v in enumerate(values):ax.text(v+(.16 if v>=0 else -.16),i,f'{v:+.2f}%',ha='left' if v>=0 else 'right',va='center',fontsize=10)
    ax.axvline(0,color='#777777',linewidth=.7)
    ax.set_title(title+'（正值更好）',fontsize=13)
    ax.set_xlabel('相对旧PhaseKV（%）');ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.set_xlim(min(values)-2.7,max(values)+2.7)
axes[0].set_yticks(range(len(cases)),[c.replace('_','  ') for c in cases],fontsize=10)
axes[0].invert_yaxis()
fig.suptitle('Prompt90对旧PhaseKV：保留解码与端到端吞吐比较',fontsize=16)
fig.text(.07,.02,'历史跨GPU观测比较，非同卡受控加速比。旧TTFT面板已移除，吞吐数据未改动。',fontsize=10)
fig.tight_layout(rect=(0,.045,1,.95))
save_chart(fig,'优化前_Prompt90/分析与图表/速度详细图表/05_新旧PhaseKV_16点变化')
plt.close(fig)

# Verify every unedited file in the snapshot remains byte-identical.
allowed={c['path'] for c in changed}
unchanged=0
for name,h in before.items():
    p=ROOT/name
    assert p.is_file(),name
    if name not in allowed:
        assert sha(p)==h,name
        unchanged+=1
for item in changed:assert sha(ROOT/item['path'])==item['after_sha256']
expected_archive='90e0fe6a2985bb6179db680cab523e9aee3240ff49d4ab1b5d4f830ab8263754'
assert sha(BASE/'a800_ttft_ab_results_20261107_215414.tar.gz')==expected_archive

# No historical TTFT keys remain in the family-only record trees.
family_dirs=['多算法最终对照/phasekv','优化前_Prompt90/质量原始记录','优化前_Prompt90/速度原始记录','优化前_Prompt90/实验身份','BetterMemory_速度与峰值显存/逐轮记录','BetterMemory_速度与峰值显存/预热记录']
for sub in family_dirs:
    for p in (BASE/sub).rglob('*.json'):
        found=[];scrub(load(p),found);assert not found,(str(p),found)
for p in [BASE/'优化前_Prompt90/分析与图表/analysis.json',BASE/'BetterMemory_速度与峰值显存/全部算法速度与峰值对比.json']:
    for m,metrics in load(p)['speed'].items():
        if family(m):found=[];scrub(metrics,found);assert not found

report={'scope':NOTICE,'changed_files':len(changed),'json_files_with_removed_items':len(removed),'removed_fields_or_cells':sum(map(len,removed.values())),'unchanged_files_verified':unchanged,'latest_archive_sha256':expected_archive,'charts':chart_paths,'changes':changed,'validation':'PASS: selected TTFT deletions only in data; all other snapshotted files byte-identical. Documentation and mixed charts updated. No previous TTFT values backed up.'}
REPORT.write_bytes(dump(report))
(AUDIT/'清理说明.txt').write_text(NOTICE+'\n\n已移除旧数值及相关图表项，未保存旧TTFT数值备份。\n操作前指纹.json只保存文件路径与SHA256，不保存数据内容。\n清理与校验.json记录字段路径、文件前后指纹及校验结果。\n此前保留来源清单的哈希是清理前历史快照，不应据此判定此次授权清理后的文件损坏。\n目录名中的“原始记录”沿用原名；PhaseKV系列旧文件现为删除TTFT字段后的记录。\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='changes'},ensure_ascii=False,indent=2))
