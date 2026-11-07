"""Rebuild extended preprint statistics from stored records (stdlib only).

This verifies recorded outputs. It does not rerun training, a simulator, or the
archived multi-axis bootstrap. verify_current.py separately checks file hashes
and all main-stream metrics.
"""
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/current'


def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
        raise AssertionError(f'{actual} != {expected}')


def verify():
    stream = read('deployment_pareto.json')
    for unit in stream['units']:
        assert len(unit['records']) == 348
        shifted = [r for r in unit['records'] if r['mechanism'] != 'source']
        assert len(shifted) == 174
        assert sum(r['measured'] for r in shifted) == 126

    control = read('closed_loop_control.json')
    assert len(control['rows']) == 450
    groups = defaultdict(list)
    pairs = {}
    for row in control['rows']:
        assert row['decisions'] == 13
        assert row['elapsed_steps'] == 104
        groups[row['arm']].append(row)
        key = row['model_seed'], row['task'], row['replica'], row['arm']
        assert key not in pairs
        pairs[key] = row['cumulative_reward']
    assert len(groups) == 6
    control_values = {}
    for arm, rows in groups.items():
        assert len(rows) == 75
        reward = mean(r['cumulative_reward'] for r in rows)
        success = mean(r['success'] for r in rows)
        close(reward, control['aggregate']['arms'][arm]['mean_reward'])
        close(success, control['aggregate']['arms'][arm]['success_rate'])
        control_values[arm] = {'reward': reward, 'success_rate': success}
    differences = [(task, pairs[(model, task, replica, 'safe_fast')] - reward)
                   for (model, task, replica, arm), reward in pairs.items()
                   if arm == 'frozen']
    assert round(mean(v for _, v in differences), 2) == 7.90
    task_gain = {task: mean(v for name, v in differences if name == task)
                 for task in sorted({t for t, _ in differences})}
    assert all(v > 0 for v in task_gain.values())

    joint = read('joint_training.json')
    assert len(joint['rows']) == 5
    extracted = defaultdict(list)
    stages = defaultdict(list)
    for row in joint['rows']:
        seq = row['baseline']['summary']
        act = row['arms']['action_only_shared_update']['evaluation']['summary']
        both = row['arms']['joint_multimodal_action']['evaluation']['summary']
        for label, values in [('sequential', seq), ('action_only', act), ('joint', both)]:
            stages[label].append(values)
        extracted['action_mse_improvement'].append(1 - both['action_correct_mse'] / seq['action_correct_mse'])
        extracted['next_slice_improvement'].append(1 - both['mean_normalized_next_slice'] / seq['mean_normalized_next_slice'])
        extracted['semantic_accuracy_change'].append(both['mean_coupled_semantic_accuracy'] - seq['mean_coupled_semantic_accuracy'])
        extracted['binding_r1_change'].append(both['binding_mean_class_r1'] - seq['binding_mean_class_r1'])
        extracted['joint_vs_action_only_action_mse_relative'].append(both['action_correct_mse'] / act['action_correct_mse'] - 1)
        extracted['joint_vs_action_only_binding_r1'].append(both['binding_mean_class_r1'] - act['binding_mean_class_r1'])
    for field, values in extracted.items():
        close(mean(values), joint['aggregate'][field]['mean'])
        for actual, stored in zip(values, joint['aggregate'][field]['values']):
            close(actual, stored)
    fields = ['action_correct_mse', 'mean_normalized_next_slice',
              'mean_coupled_semantic_accuracy', 'binding_mean_class_r1']
    joint_stages = {stage: {k: mean(r[k] for r in rows) for k in fields}
                    for stage, rows in stages.items()}
    assert len(read('causal_closure.json')['traces']) == 36
    print(json.dumps({'status': 'PASS', 'control': control_values,
                      'task_reward_gains': task_gain, 'joint_stages': joint_stages,
                      'scope': 'Recorded-output verification; no fresh runs or CI resampling.'},
                     indent=2))


if __name__ == '__main__':
    verify()
