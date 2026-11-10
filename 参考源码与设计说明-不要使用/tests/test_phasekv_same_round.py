import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_phasekv_same_round as r


class TestSameRound(unittest.TestCase):
    def test_configs_and_new_documents(self):
        plan=r.configs()
        self.assertEqual([x[0] for x in plan],['Full-native','Full-matched','KIVI','SnapKV','SCOPE','PhaseKV'])
        self.assertEqual(plan[-1][1],r.s.spec(.7,4,128,4,1.))
        self.assertTrue(all(policy=='matched' for _,_,policy in plan[1:]))
        args=r.arguments([]);settings,_=r.settings(args)
        self.assertEqual(len(settings['performance_shapes']),9)
        for task in settings['tasks']:
            old=set(r.b.sample_indices(200,72,args.seed,task))
            new=set(r.b.sample_indices(200,32,args.seed,task,args.sample_offset))
            self.assertFalse(old & new)

    def test_controller_pairs_all_methods_resumes_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory(dir=r.b.ROOT,prefix='_same_round_test_') as tmp:
            root=Path(tmp);out=root/'out';data=root/'longbench'
            for task in r.s.SCREEN_TASKS:
                path=data/'data'/(task+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('{}\n'*120,encoding='utf-8')
            options=['--out',str(out),'--data-dir',str(data),'--quick','--samples','1','--repeats','1']
            launched=[]
            def execute(command,log):
                if '__identity__' in command:
                    profile=r.t.read(command[-2])[0]
                    r.b.atomic_json(command[-1],{profile['name']:{'resolved_path':'unused'}})
                elif '__validate__' in command:
                    r.b.atomic_json(command[-1],{'status':'passed'})
                elif '__worker__' in command:
                    job=r.t.read(command[-1]);name=Path(job['records_path']).parent.name;launched.append(name)
                    if job['spec']['method']!='full':
                        self.assertEqual(job['matched_baseline'],str(out/'runs/Full-matched/records.jsonl'))
                    rows=[]
                    for key in sorted(r.s.expected_keys(job['args'],job['args']['data_dir'])):
                        parts=key.split('/');native=job['prefill_policy']=='native'
                        row=dict(case_id=key,protocol_hash=job['protocol_hash'],parameters=job['spec'],
                            method=job['spec']['method'],status='ok',input_hash=key,generated_ids=[2 if native else 1],
                            prefill_logits_sha256='native' if native else 'matched',prefill_policy=job['prefill_policy'],
                            phase=parts[0],answers=None)
                        if parts[0]=='quality':
                            row.update(task=parts[1],length_budget=int(parts[2]),sample_index=int(parts[3]),score=40.,answers=['x'])
                        else:
                            row.update(sweep=parts[1],input_tokens=int(parts[2]),output_tokens=int(parts[3]),requested_output_tokens=int(parts[3]),
                                decode_tokens_per_s=40.,final_kv_bytes=1000,prefill_kv_bytes=500,total_s=1.,prefill_and_compress_s=.1,peak_allocated_bytes=10000)
                        rows.append(row)
                    Path(job['records_path']).write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            with patch.object(r.s,'execute',side_effect=execute),patch.object(r.b,'check_sources'),patch.object(r.b,'prepare_data',return_value={}),patch.object(r.b,'source_hashes',return_value={}),contextlib.redirect_stdout(io.StringIO()):
                r.main(options)
                self.assertEqual(launched,[name for name,_,_ in r.configs()])
                result=r.t.read(out/'comparison.json')
                self.assertEqual(len(result['results']),6)
                self.assertEqual(set(result['phasekv_vs_methods']),{'KIVI','SnapKV','SCOPE'})
                for name in launched:
                    metrics=result['results'][name]['comparisons']
                    self.assertEqual(len(metrics),2)
                    self.assertTrue(metrics['Full-matched']['valid'])
                    self.assertEqual(len(metrics['Full-matched']['shapes']),9)
                self.assertEqual(result['prefill_audit']['KIVI']['against']['Full-matched']['logits_hash_differences'],0)
                self.assertEqual(result['prefill_audit']['PhaseKV']['against']['Full-native']['first_token_differences'],8)
                r.main(options+['--resume'])
                self.assertEqual(len(launched),6)
                with self.assertRaisesRegex(ValueError,'changed'):
                    r.main(options+['--resume','--sample-offset','73'])

            manifest=r.t.read(out/'manifest.json')
            folder=out/'runs/SCOPE';rp=folder/'records.jsonl'
            rows=r.b.read_records(rp);rows[0]['prefill_logits_sha256']='wrong'
            def save():
                rp.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
                done=r.t.read(folder/'complete.json');done['hash']=r.b.file_hash(rp)
                r.b.atomic_json(folder/'complete.json',done)
            save()
            with self.assertRaisesRegex(ValueError,'logits differ'):
                r.report(out,manifest)
            # Real OOM rows have no logits/policy: report them as invalid, not as a protocol crash.
            rows[0]['status']='oom'
            rows[0].pop('prefill_policy');rows[0].pop('prefill_logits_sha256')
            save()
            report=r.report(out,manifest)
            self.assertFalse(report['SCOPE']['comparisons']['Full-matched']['valid'])
            self.assertFalse(r.t.read(out/'comparison.json')['phasekv_vs_methods']['SCOPE']['valid'])
            self.assertIsNotNone(r.s.label(r.s.spec()))


if __name__=='__main__':
    unittest.main()
