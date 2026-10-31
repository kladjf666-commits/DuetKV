import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import retest_phasekv_prompt as r


class TestRetest(unittest.TestCase):
    def test_native_validation_preserves_required_attention_oracle(self):
        original=r.b.prefill_model
        def validator(profiles,report_path):
            self.assertIs(r.b.prefill_model,original)
            return [{'status':'passed'}]
        with tempfile.TemporaryDirectory(dir=r.b.ROOT,prefix='_retest_test_') as tmp:
            path=Path(tmp)/'validation.json'
            with patch.object(r.b,'gpu_validation',side_effect=validator):
                r.validate_job({'prefill_policy':'native','profiles':[]},str(path))
            self.assertEqual(r.read(path)['status'],'passed')

    def test_six_configs_force_factorial_and_holdout(self):
        configs=r.configs()
        self.assertEqual(len(configs),8)
        self.assertEqual({(s['prefill_keep_ratio'],s['prompt_bits']) for _,s,_ in configs[2:]},
                         {(ratio,bits) for ratio in (.7,1.) for bits in (2,4,8)})
        self.assertTrue(all(s['recent_window']==128 and s['cold_bits']==4 and s['decode_keep_ratio']==1 for _,s,_ in configs[2:]))
        args=r.arguments([])
        a,_=r.settings(args)
        for task in a['tasks']:
            old=set(r.b.sample_indices(200,40,args.seed,task))
            new=set(r.b.sample_indices(200,args.samples,args.seed,task,args.sample_offset))
            self.assertFalse(old & new)

    def test_full_logits_must_match_even_when_first_token_matches(self):
        ref=dict(status='ok',phase='quality',input_hash='same',prefill_logits_sha256='a',generated_ids=[1,2])
        r.verify_pair('q',dict(ref),ref)
        with self.assertRaisesRegex(ValueError,'logits differ'):
            r.verify_pair('q',dict(ref,prefill_logits_sha256='b'),ref)
        with self.assertRaisesRegex(ValueError,'Missing'):
            r.verify_pair('q',ref,None)
        with self.assertRaisesRegex(ValueError,'input/status'):
            r.verify_pair('q',dict(ref,input_hash='other'),ref)

    def test_policy_restores_hooks_and_native_does_not_force_scores(self):
        import torch
        calls=[]
        def prefill(model,ids,chunk,need_scores,observation=32):
            calls.append(need_scores)
            return types.SimpleNamespace(logits=torch.tensor([[[1.,2.]]],dtype=torch.bfloat16))
        def generate():
            r.b.prefill_model(None,None,512,False)
            return {'generated_ids':[1]}
        with patch.object(r.b,'prefill_model',prefill),patch.object(r.b,'generate',generate):
            recorder=r.b.Recorder
            with r.worker_policy('matched'):
                value=r.b.generate()
                self.assertEqual(value['prefill_logits_sha256'],r.fingerprint(torch.tensor([[1.,2.]],dtype=torch.bfloat16)))
                self.assertEqual(value['prefill_policy'],'matched')
            self.assertIs(r.b.generate,generate)
            self.assertIs(r.b.Recorder,recorder)
            with r.worker_policy('native'):
                r.b.generate()
            self.assertEqual(calls,[True,False])
            with self.assertRaisesRegex(RuntimeError,'probe'):
                with r.worker_policy('matched'):
                    raise RuntimeError('probe')
            self.assertIs(r.b.prefill_model,prefill)

    def test_fingerprint_distinguishes_dtype_and_values(self):
        import torch
        x=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
        self.assertEqual(r.fingerprint(x),r.fingerprint(x.clone()))
        self.assertNotEqual(r.fingerprint(x),r.fingerprint(x.float()))
        self.assertNotEqual(r.fingerprint(x),r.fingerprint(x+1))

    def test_controller_eight_jobs_resume_and_both_baselines(self):
        with tempfile.TemporaryDirectory(dir=r.b.ROOT,prefix='_retest_test_') as tmp:
            root=Path(tmp);out=root/'out';data=root/'longbench'
            for task in r.s.SCREEN_TASKS:
                path=data/'data'/(task+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('{}\n'*72,encoding='utf-8')
            options=['--out',str(out),'--data-dir',str(data),'--quick','--samples','1','--repeats','1']
            launched=[]
            def execute(command,log):
                if '__identity__' in command:
                    profile=r.read(command[-2])[0]
                    r.b.atomic_json(command[-1],{profile['name']:{'resolved_path':'unused'}})
                elif '__validate__' in command:
                    r.b.atomic_json(command[-1],{'status':'passed'})
                elif '__worker__' in command:
                    job=r.read(command[-1]);name=Path(job['records_path']).parent.name;launched.append(name)
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
                result=r.read(out/'comparison.json')
                self.assertEqual(len(result['rows']),8)
                self.assertEqual(len(result['rows'][-1]['comparisons']),2)
                self.assertEqual(result['first_token_differences'][launched[-1]]['Full-matched'],0)
                self.assertEqual(result['first_token_differences'][launched[-1]]['Full-native'],8)
                self.assertTrue((out/'qualified_quality.json').exists())
                r.main(options+['--resume'])
                self.assertEqual(len(launched),8)
                with self.assertRaisesRegex(ValueError,'changed'):
                    r.main(options+['--resume','--sample-offset','41'])


if __name__=='__main__':
    unittest.main()
