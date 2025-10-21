import argparse
import csv
import json
import os
import numpy as np


def read_last_metrics(log_path):
  if not os.path.exists(log_path):
    return None
  with open(log_path, 'r') as f:
    rows = list(csv.reader(f))
  if not rows:
    return None
  last = rows[-1]
  # Format: iteration,<it>, loss,<val>, accuracy,<val>
  try:
    iteration = int(float(last[1]))
    loss = float(last[3])
    acc = float(last[5])
    return { 'iteration': iteration, 'loss': loss, 'accuracy': acc }
  except Exception:
    return None


def compute_density(mask_path):
  arr = np.load(mask_path)
  density = float(arr.mean())
  return density, 1.0 - density


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--root', required=True, help='Root experiments directory')
  parser.add_argument('--out', required=True, help='Output CSV file')
  parser.add_argument('--json', default='', help='Optional JSON output path')
  args = parser.parse_args()

  rows = []
  for trial in sorted(os.listdir(args.root)):
    trial_dir = os.path.join(args.root, trial)
    if not os.path.isdir(trial_dir):
      continue
    for level in sorted((d for d in os.listdir(trial_dir) if d.isdigit()), key=int):
      run_dir = os.path.join(trial_dir, level, 'same_init')
      test_log = os.path.join(run_dir, 'test.log')
      train_log = os.path.join(run_dir, 'train.log')
      metrics_test = read_last_metrics(test_log) or {}
      metrics_train = read_last_metrics(train_log) or {}

      # Compute sparsity from masks (average across layers)
      masks_dir = os.path.join(run_dir, 'masks')
      density = sparsity = None
      if os.path.isdir(masks_dir):
        densities = []
        for f in os.listdir(masks_dir):
          if f.endswith('.npy'):
            d, s = compute_density(os.path.join(masks_dir, f))
            densities.append(d)
        if densities:
          density = float(np.mean(densities))
          sparsity = 1.0 - density

      rows.append({
        'trial': trial,
        'level': int(level),
        'density': density,
        'sparsity': sparsity,
        'test_iteration': metrics_test.get('iteration'),
        'test_loss': metrics_test.get('loss'),
        'test_accuracy': metrics_test.get('accuracy'),
        'train_iteration': metrics_train.get('iteration'),
        'train_loss': metrics_train.get('loss'),
        'train_accuracy': metrics_train.get('accuracy'),
      })

  # Write CSV
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  keys = ['trial', 'level', 'density', 'sparsity', 'test_iteration', 'test_loss', 'test_accuracy', 'train_iteration', 'train_loss', 'train_accuracy']
  with open(args.out, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
      w.writerow(r)

  # Optional JSON
  if args.json:
    with open(args.json, 'w') as jf:
      json.dump(rows, jf, indent=2)


if __name__ == '__main__':
  main()


