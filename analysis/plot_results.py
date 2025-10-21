import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_csv(path):
  rows = []
  with open(path, 'r') as f:
    r = csv.DictReader(f)
    for row in r:
      # cast types if present
      for k in ['level', 'density', 'sparsity', 'test_accuracy', 'train_accuracy']:
        if row.get(k) not in (None, ''):
          row[k] = float(row[k]) if k != 'level' else int(float(row[k]))
        else:
          row[k] = None
      rows.append(row)
  return rows


def plot_sparsity_accuracy(rows, outdir):
  os.makedirs(outdir, exist_ok=True)
  # Aggregate by level across trials
  accs = defaultdict(list)
  sparsities = {}
  for r in rows:
    if r['test_accuracy'] is not None and r['sparsity'] is not None:
      accs[r['level']].append(r['test_accuracy'])
      sparsities[r['level']] = r['sparsity']

  xs, ys_mean, ys_std = [], [], []
  for level in sorted(accs.keys()):
    vals = accs[level]
    xs.append(sparsities[level])
    ys_mean.append(sum(vals) / len(vals))
    ys_std.append((sum((v - ys_mean[-1]) ** 2 for v in vals) / len(vals)) ** 0.5)

  plt.figure()
  plt.errorbar(xs, ys_mean, yerr=ys_std, fmt='-o')
  plt.xlabel('Sparsity (1 - density)')
  plt.ylabel('Test accuracy')
  plt.title('Sparsity vs Test Accuracy (same_init)')
  plt.grid(True)
  out_path = os.path.join(outdir, 'sparsity_vs_accuracy.png')
  plt.savefig(out_path, bbox_inches='tight')
  plt.close()


def plot_per_trial(rows, outdir):
  os.makedirs(outdir, exist_ok=True)
  by_trial = defaultdict(list)
  for r in rows:
    by_trial[r['trial']].append(r)

  for trial, trs in by_trial.items():
    trs = sorted(trs, key=lambda x: x['level'])
    xs = [t['level'] for t in trs if t['test_accuracy'] is not None]
    ys = [t['test_accuracy'] for t in trs if t['test_accuracy'] is not None]
    plt.figure()
    plt.plot(xs, ys, '-o')
    plt.xlabel('Pruning level')
    plt.ylabel('Test accuracy')
    plt.title(f'Trial {trial} - Test Accuracy by Level')
    plt.grid(True)
    out_path = os.path.join(outdir, f'trial_{trial}_accuracy_by_level.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--csv', required=True, help='Parsed results CSV path')
  parser.add_argument('--outdir', required=True, help='Output plots directory')
  args = parser.parse_args()

  rows = read_csv(args.csv)
  plot_sparsity_accuracy(rows, args.outdir)
  plot_per_trial(rows, args.outdir)


if __name__ == '__main__':
  main()


