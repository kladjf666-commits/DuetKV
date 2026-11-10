"""Read-only audit of the downloaded long-output A/B; write a separate report."""
import html
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import quick_phasekv_long_output as q

ROOT=q.ROOT
SOURCE=ROOT/'results/downloads/20261027_long_output_p5/results/phasekv_long_output_quick_001'
OUT=ROOT/'reports/phasekv_long_output_20261027'


def main():
    read=q.t.read
    manifest=read(SOURCE/'manifest.json')
    assert q.b.digest({k:v for k,v in manifest.items() if k!='hash'})==manifest['hash']
    assert read(SOURCE/'complete.json')=={'status':'complete','manifest_hash':manifest['hash']}
    saved=read(SOURCE/'summary.json')
    assert saved['manifest_hash']==manifest['hash']
    guard=read(SOURCE/'gpu_guard.json')
    assert guard['status']=='passed' and len(guard['checks'])==4
    all_rows={stage:q.load_rows(SOURCE,manifest,stage) for stage in ('baseline','p5')}
    for stage in all_rows:
        assert q.b.file_hash(SOURCE/stage/'records.jsonl')==saved['record_hashes'][stage]
    refs={r['case_id']:r for r in all_rows['baseline']}
    for r in all_rows['p5']:
        q.t.verify_pair(r['case_id'],r,refs[r['case_id']])
    measured={stage:[r for r in rr if not r['is_warmup']] for stage,rr in all_rows.items()}
    assert all(len(rr)==1 for rr in measured.values()),'This report describes a single-repeat run'
    a,z=measured['baseline'][0],measured['p5'][0]
    assert a['input_tokens']==z['input_tokens']==1024
    assert a['output_tokens']==z['output_tokens']==4096
    for stage,rr in measured.items():
        original=saved['shapes'][0]['stages'][stage]
        assert original['decode_tokens_per_s']==st.median(r['decode_tokens_per_s'] for r in rr)
        assert original['kv_mib']==st.median(r['final_kv_bytes'] for r in rr)/2**20
    gain=z['decode_tokens_per_s']/a['decode_tokens_per_s']-1
    assert abs(gain*100-saved['shapes'][0]['throughput_change_percent'])<1e-10
    p5=z['storage_evidence'];base=a['storage_evidence']
    extra=z['final_kv_bytes']-a['final_kv_bytes']
    assert extra==p5['page_free_bytes']
    assert p5['code_payload_bytes']==base['code_bytes']
    assert p5['metadata_payload_bytes']==base['metadata_bytes']
    difference=next((i for i,(x,y) in enumerate(zip(a['generated_ids'],z['generated_ids'])) if x!=y),None)
    env={stage:read(next((SOURCE/stage).glob('records.environment.*.json'))) for stage in all_rows}
    environment_keys=('gpu','torch','cuda','triton','transformers','external_bytes_at_start')
    assert all(env['baseline'][k]==env['p5'][k] for k in environment_keys)
    stats=[('解码吞吐（token/s）',f"{a['decode_tokens_per_s']:.2f}",f"{z['decode_tokens_per_s']:.2f}",f'+{gain*100:.2f}%'),
        ('解码耗时（秒）',f"{a['decode_s']:.2f}",f"{z['decode_s']:.2f}",f"减少{a['decode_s']-z['decode_s']:.2f}秒"),
        ('Prefill＋压缩（秒）',f"{a['prefill_and_compress_s']:.3f}",f"{z['prefill_and_compress_s']:.3f}",'单次测量'),
        ('总耗时（秒）',f"{a['total_s']:.2f}",f"{z['total_s']:.2f}",f"减少{100*(1-z['total_s']/a['total_s']):.2f}%"),
        ('实际KV容量（MiB）',f"{a['final_kv_bytes']/2**20:.2f}",f"{z['final_kv_bytes']/2**20:.2f}",f'+{extra/2**20:.2f} MiB'),
        ('峰值allocated（GiB）',f"{a['peak_allocated_bytes']/2**30:.3f}",f"{z['peak_allocated_bytes']/2**30:.3f}",'包括权重及临时内存'),
        ('冷却触发次数',a['decode_compression_steps'],z['decode_compression_steps'],'相同')]
    paragraphs=[
        f'本次单次正式测试中，P5解码吞吐提升{gain*100:.2f}%，总耗时减少约{a["total_s"]-z["total_s"]:.2f}秒。可以确认这次运行更快，不能据此认定所有场景都稳定提速。',
        '配置为Qwen2.5-7B-Instruct、4090 D、输入1024、强制输出4096；两版均为70%保留/4-bit、热区128、冷区4-bit、生成保留100%、lazy_g16。P5为累积优化，cold_group8、page_slots32，未开启独立GQA复用。',
        '每版1次预热、1次正式测量；主表只统计正式测量。数据完整性、manifest、完成标记、记录哈希、参数、输入、prefill logits和首token配对均通过检查；无失败或提前停止的记录。',
        f'目标GPU检查覆盖cold=0/7/31/247的打包码/元数据、attention、环绕与增长，全部通过；最大attention绝对差为{max(c["max_abs_error"] for c in guard["checks"]):.8f}。这不等于全部GPU回归或准确率验收通过。',
        f'实际KV增加{extra/2**20:.3f} MiB，恰好对应页式存储预留的空闲容量；有效码与元数据字节数和优化前一致。P5另报告辅助工作区{p5["auxiliary_bytes"]/2**20:.2f} MiB；基线未单列同口径辅助字节，不能据此计算辅助内存节省。',
        f'两版生成序列从第{difference+1}个token开始分叉（从1计数）。输入、首token和生成长度一致，但完整输出不同，因此不能声称质量不变或逐步输出完全等价。当前数据只评价吞吐。',
        'P5预热为34.27 token/s，正式为37.05 token/s；预热含可能的编译/缓存和设备状态影响，不能并入正式平均，也不能作为第二次独立重复。',
        '本轮只有baseline→P5的顺序执行，没有交替顺序或足够重复；无法拆分P1—P5各项收益，也不能用历史Full速度作严格同轮对照。建议同条件至少3次正式重复，另起目录，再检查长输入＋4096输出。']
    payload=dict(source=SOURCE.relative_to(ROOT).as_posix(),audit='passed',manifest_hash=manifest['hash'],
        record_hashes=saved['record_hashes'],config=manifest['groups'],metrics=saved['shapes'],
        first_different_token_1based=difference+1,extra_kv_equals_page_free=True,gpu_guard=guard,
        environment={k:env['p5'][k] for k in environment_keys},conclusions=paragraphs)
    OUT.mkdir(parents=True,exist_ok=True)
    q.b.atomic_json(OUT/'analysis.json',payload)
    (OUT/'analysis.txt').write_text('PhaseKV P5 长输出快速验证\n\n'+'\n'.join(' | '.join(map(str,row)) for row in [('指标','优化前','P5','变化'),*stats])+'\n\n'+'\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    table='<table><thead><tr>'+''.join('<th>'+x+'</th>' for x in ('指标','优化前','P5','变化'))+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in stats)+'</tbody></table>'
    content='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PhaseKV P5 长输出快速验证</title><style>body{font:16px system-ui;line-height:1.75;max-width:1050px;margin:35px auto;padding:20px;color:#203249;background:#f7fafc}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd8e3;padding:10px;text-align:left}th{background:#e6eff7}.lead{background:#e4f2ed;padding:18px;border-radius:10px}.scroll{overflow:auto}a{color:#175bb0}</style><h1>PhaseKV P5：长输出快速验证</h1><p>2026-10-27 · 1024输入＋4096输出 · 单次正式测量</p><p class="lead">'+html.escape(paragraphs[0])+'</p><div class="scroll">'+table+'</div>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs[1:])+'<p><a href="analysis.json">审计与明细JSON</a> · <a href="../../results/downloads/20261027_long_output_p5/results/phasekv_long_output_quick_001/summary.txt">原始测速汇总</a> · <a href="../../PROJECT_GUIDE.txt">项目目录导航</a></p></html>'
    (OUT/'report.html').write_text(content,encoding='utf-8')
    print(OUT/'report.html')


if __name__=='__main__':main()
