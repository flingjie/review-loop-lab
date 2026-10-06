import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import loop

class LoopTests(unittest.TestCase):
    def test_fixtures(self):
        loop.verify_fixtures()

    def test_silence_is_not_improvement(self):
        old = dict(fp=2, fn=0, errors=0)
        self.assertFalse(loop.better(dict(fp=0,fn=3,errors=0), old))
        self.assertTrue(loop.better(dict(fp=1,fn=0,errors=0), old))
        self.assertFalse(loop.better(old, old))
        self.assertFalse(loop.better(dict(fp=0,fn=0,errors=1), old))

    def test_wrong_type_counts_as_fp_and_fn(self):
        m = loop.score([{'expected_type':'boundary','response':{'findings':[{'kind':'wrong_status'}]}}])
        self.assertEqual((m['tp'],m['fp'],m['fn']), (0,1,1))

    def test_invalid_output_not_success(self):
        m = loop.score([{'expected_type':'boundary','error':'timeout'}, {'expected_type':None,'error':'bad JSON'}])
        self.assertEqual((m['fn'], m['errors']), (1,2))

    def test_public_payload_excludes_answers(self):
        for c in loop.load_cases():
            data = loop.public_case(c)
            self.assertEqual(set(data), {'requirement','before','after_numbered','diff'})

    def test_schema(self):
        c = loop.load_cases()[0]
        for obj in [{'findings':[{'kind':'boundary','line':True,'trigger':'a','consequence':'b'}]}, {'findings':[], 'extra':1}]:
            with self.assertRaises(ValueError):
                loop.validate_review(obj,c)

    def test_demo_full_loop(self):
        import argparse
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)/'run'
            args = argparse.Namespace(mode='demo',out=str(out),max_calls=250,timeout=10,rounds=5,holdout_repeats=3)
            loop.run(args)
            s = json.loads((out/'summary.json').read_text())
            self.assertEqual([r['accepted'] for r in s['rounds']], [True,False,False,False,False])
            self.assertEqual(s['calls'], 221)
            self.assertEqual(s['holdout']['best'][0]['fp'],0)
            self.assertIn('DEMO', (out/'report.md').read_text())
            # Optimizer messages receive revision examples only.
            events = [json.loads(x) for x in (out/'calls.jsonl').read_text().splitlines()]
            for event in events:
                if 'optimizer' in event['tag']:
                    data = json.loads(event['user'])
                    requirements = {c['requirement'] for c in loop.load_cases() if c['split']=='revision'}
                    self.assertTrue(all(x['case']['requirement'] in requirements for x in data['failures']+data['correct_examples']))

    def test_live_wire_format_with_fake_http(self):
        import io
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ', {'LLM_API_KEY':'test-key', 'LLM_MODEL':'test-model'}):
            client=loop.Client('live', Path(td), 3, 10)
            body={'choices':[{'message':{'content':'{"findings":[]}'},'finish_reason':'stop'}],'usage':{'total_tokens':10}}
            with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(body).encode())) as call:
                self.assertEqual(client.ask('JSON','input','test'), {'findings':[]})
                req=call.call_args.args[0]
                self.assertTrue(req.full_url.endswith('/chat/completions'))
                self.assertEqual(json.loads(req.data)['response_format'], {'type':'json_object'})
            self.assertEqual(client.tokens,10)
            self.assertNotIn('test-key',(Path(td)/'calls.jsonl').read_text())

    def test_budget_aborts(self):
        with tempfile.TemporaryDirectory() as td:
            client=loop.Client('demo',Path(td),1,10)
            client.ask('','', 'optimizer',round_id=1)
            with self.assertRaises(loop.CallFailure):
                client.ask('','', 'optimizer',round_id=2)

if __name__=='__main__':
    unittest.main()
