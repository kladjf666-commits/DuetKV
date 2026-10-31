import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import quick_phasekv_long_output as q


class TestQuickLongOutput(unittest.TestCase):
    def test_plan_has_no_quality_and_uses_frozen_baseline(self):
        p=q.plan(q.arguments([]))
        self.assertEqual(p['settings']['phases'],['performance'])
        self.assertEqual(p['settings']['tasks'],[])
        self.assertEqual(p['settings']['performance_shapes'],[(1024,4096,'long_output_quick')])
        self.assertEqual([k for k,_ in p['groups']],['baseline','p5'])
        self.assertTrue(p['groups'][0][1]['saved_baseline'])
        self.assertTrue(p['groups'][1][1]['optimizations']['hot_ring'])
        self.assertFalse(p['groups'][1][1]['optimizations']['gqa_reuse'])
        self.assertEqual(p['generations_per_group'],4)
        with self.assertRaisesRegex(ValueError,'context limit'):
            q.plan(q.arguments(['--prompts','32000']))

    def test_controller_and_summary_reject_incomplete_or_unaudited_runs(self):
        with tempfile.TemporaryDirectory(dir=q.ROOT,prefix='_quick_long_test_') as tmp:
            out=Path(tmp)/'out';launched=[]
            def execute(command,log):
                if '__guard__' in command:
                    q.b.atomic_json(command[-1],{'status':'passed'})
                elif '__identity__' in command:
                    profile=q.t.read(command[-2])[0]
                    q.b.atomic_json(command[-1],{profile['name']:{'resolved_path':'unused'}})
                elif '__worker__' in command:
                    job=q.t.read(command[-1]);stage=Path(job['records_path']).parent.name
                    launched.append(stage)
                    if stage=='p5':
                        self.assertEqual(job['matched_baseline'],str(out/'baseline/records.jsonl'))
                    rows=[]
                    for n,m,sweep in job['args']['performance_shapes']:
                        for i in range(job['args']['repeats']+1):
                            rows.append(dict(case_id=f'performance/{sweep}/{n}/{m}/{i}',
                                status='ok',parameters=job['spec'],protocol_hash=job['protocol_hash'],
                                prefill_policy='matched',input_hash='input',prefill_logits_sha256='logits',
                                generated_ids=[1],fixed_output=True,input_tokens=n,output_tokens=m,
                                requested_output_tokens=m,is_warmup=i==0,decode_compression_steps=247,
                                decode_tokens_per_s=999 if i==0 else (20 if stage=='baseline' else 30),
                                decode_s=200,total_s=210 if stage=='baseline' else 140,
                                prefill_and_compress_s=10,final_kv_bytes=1000,peak_allocated_bytes=2000))
                    Path(job['records_path']).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
            with patch.object(q.opt.same.s,'execute',side_effect=execute),patch.object(q.b,'source_hashes',return_value={}),patch.object(q.opt,'configure'),contextlib.redirect_stdout(io.StringIO()):
                q.main(['--out',str(out),'--repeats','1'])
                self.assertEqual(launched,['baseline','p5'])
                summary=q.t.read(out/'summary.json')['shapes'][0]
                self.assertEqual(summary['decode_speedup'],1.5)
                self.assertEqual(summary['throughput_change_percent'],50)
                self.assertEqual(summary['stages']['baseline']['samples_tokens_per_s'],[20])
                self.assertTrue((out/'complete.json').exists())
                with self.assertRaises(FileExistsError):q.main(['--out',str(out)])
                path=out/'p5/records.jsonl';raw=path.read_text(encoding='utf-8')
                rows=q.b.read_records(path)
                def write():path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
                rows[1]['prefill_logits_sha256']='bad';write()
                with self.assertRaisesRegex(ValueError,'logits differ'):q.summarize(out)
                path.write_text(raw,encoding='utf-8');rows=q.b.read_records(path)
                rows[1]['status']='oom';write()
                with self.assertRaisesRegex(ValueError,'Failed case'):q.summarize(out)
                path.write_text(raw,encoding='utf-8');rows=q.b.read_records(path)
                rows.pop();write()
                with self.assertRaisesRegex(ValueError,'Missing or duplicate'):q.summarize(out)


if __name__=='__main__':unittest.main()
