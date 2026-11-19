import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('behavior_runner', Path(__file__).resolve().parents[1] / 'scripts/behavior/run.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def rows(size=16, families=(0, 1, 2)):
    return [{'id': f'{family}-{i}', 'task': f'task{family}', 'family_index': family,
             'input_ids': [3, 4], 'mask': [True, True], 'positions': [0, 1],
             'expected_id': 5, 'foil_id': 6, 'top1_id': 5 if i < 13 else 7}
            for family in families for i in range(size)]


class BehaviorTests(unittest.TestCase):
    def test_reject_nonfinite_outside_label_tokens(self):
        import torch
        for bad in (float('-inf'), float('inf'), float('nan')):
            # Expected and foil entries remain finite; invalid unlabelled entry must fail.
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'Nonfinite vocabulary'):
                r.require_finite_logits(torch.tensor([1.0, 0.0, bad]))
        r.require_finite_logits(torch.tensor([1.0, 0.0, -3.0]))

    def test_screen_threshold_and_deterministic_pair(self):
        result = r.summarize(rows(), 'screen')
        self.assertEqual(result['selected'], [0, 1])
        self.assertEqual(result['verdict'], 'QUALIFIED')
        self.assertEqual([g['correct'] for g in result['families']], [13, 13, 13])

    def test_full_vocabulary_correctness_not_foil_win(self):
        data = rows()
        for row in data:
            if row['family_index'] != 2:
                row.update(top1_id=7, expected_logit=10, foil_logit=0)
        self.assertEqual(r.summarize(data, 'screen')['verdict'], 'NO_QUALIFY')
        self.assertEqual(r.summarize(data, 'screen')['selected'], [])

    def test_qualification_threshold_both_families(self):
        data = rows(32, (0, 2))
        for row in data:
            row['top1_id'] = 5 if int(row['id'].split('-')[1]) < 26 else 7
        self.assertEqual(r.summarize(data, 'qualify')['verdict'], 'QUALIFIED')
        data[0]['top1_id'] = 7
        self.assertEqual(r.summarize(data, 'qualify')['verdict'], 'NO_QUALIFY')

    def test_exact_fixture_balance(self):
        self.assertEqual(r.validate_rows({'split': 'screen', 'rows': rows()}, 'screen'), 96)
        self.assertEqual(r.validate_rows({'split': 'qualify', 'rows': rows(32, (0, 2))}, 'qualify', [0, 2]), 128)
        with self.assertRaises(ValueError):
            r.validate_rows({'split': 'screen', 'rows': rows()[:-1]}, 'screen')

    def test_reject_padding_duplicate_and_wrong_family(self):
        for mutation in ('padding', 'duplicate', 'family'):
            data = rows()
            if mutation == 'padding':
                data[0]['mask'][0] = False
            elif mutation == 'duplicate':
                data[0]['id'] = data[1]['id']
            else:
                data[0]['family_index'] = 9
            with self.assertRaises(ValueError):
                r.validate_rows({'split': 'screen', 'rows': data}, 'screen')


if __name__ == '__main__':
    unittest.main()
