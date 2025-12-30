import torch
import os
import numpy as np
from pathlib import Path
import deepcore.datasets as datasets

# Find the latest checkpoint
result_dir = Path('result')
ckpt_files = list(result_dir.glob('*.ckpt'))

if not ckpt_files:
    print("No checkpoint files found in result/")
    exit(1)

# Get the latest checkpoint by modification time
latest = max(ckpt_files, key=lambda x: x.stat().st_mtime)
print(f'Loading checkpoint: {latest.name}')
print(f'File path: {latest}')
print()

# Load checkpoint (weights_only=False for compatibility with older checkpoints)
ckpt = torch.load(latest, map_location='cpu', weights_only=False)

print('=== Checkpoint Contents ===')
print(f'Keys: {list(ckpt.keys())}')
print()

# Print experiment info
if 'exp' in ckpt:
    print(f'Experiment: {ckpt["exp"]}')
if 'epoch' in ckpt:
    print(f'Epoch: {ckpt["epoch"]}')
if 'best_acc1' in ckpt:
    print(f'Best Accuracy: {ckpt["best_acc1"]:.4f}%')
print()

# Print coreset information
if 'subset' in ckpt:
    subset = ckpt['subset']
    print('=== Coreset Information ===')
    print(f'Number of samples in coreset: {len(subset["indices"])}')
    
    indices = subset['indices']
    if hasattr(indices, 'shape'):
        print(f'Indices shape: {indices.shape}')
        print(f'Indices dtype: {indices.dtype}')
    else:
        print(f'Indices type: {type(indices)}')
        print(f'Indices length: {len(indices)}')
    
    print(f'Subset keys: {list(subset.keys())}')
    
    if 'weights' in subset:
        weights = subset['weights']
        if hasattr(weights, 'shape'):
            print(f'Has weights: Yes (shape: {weights.shape}, dtype: {weights.dtype})')
        else:
            print(f'Has weights: Yes (type: {type(weights)})')
    else:
        print('Has weights: No')
    
    # Show first few indices
    if len(indices) > 0:
        print(f'\nFirst 10 indices: {indices[:10] if len(indices) >= 10 else indices}')
        if len(indices) > 10:
            print(f'Last 10 indices: {indices[-10:]}')
else:
    print('No subset information found in checkpoint')

# Print selection arguments if available
if 'sel_args' in ckpt:
    print('\n=== Selection Arguments ===')
    for key, value in ckpt['sel_args'].items():
        print(f'{key}: {value}')

# Print class to label distribution
if 'subset' in ckpt:
    subset = ckpt['subset']
    indices = subset['indices']
    
    # Try to get dataset name from checkpoint filename or use default
    dataset_name = 'MNIST'  # Default, could be extracted from filename
    if 'MNIST' in latest.name:
        dataset_name = 'MNIST'
    elif 'CIFAR10' in latest.name:
        dataset_name = 'CIFAR10'
    elif 'CIFAR100' in latest.name:
        dataset_name = 'CIFAR100'
    
    print(f'\n=== Class to Label Distribution ===')
    print(f'Loading {dataset_name} dataset to get labels...')
    
    try:
        # Load the dataset
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[dataset_name]('./data')
        
        # Get labels for the selected indices
        if hasattr(dst_train, 'targets'):
            all_labels = np.array(dst_train.targets)
        elif hasattr(dst_train, 'labels'):
            all_labels = np.array(dst_train.labels)
        else:
            # Fallback: get labels by indexing
            all_labels = np.array([dst_train[i][1] for i in range(len(dst_train))])
        
        # Get labels for selected indices
        selected_labels = all_labels[indices]
        
        # Count distribution
        unique_labels, counts = np.unique(selected_labels, return_counts=True)
        
        print(f'\nTotal samples: {len(selected_labels)}')
        print(f'Number of classes: {len(unique_labels)}')
        print(f'\nClass Distribution:')
        print('-' * 50)
        print(f'{"Class":<10} {"Label":<10} {"Count":<10} {"Percentage":<10}')
        print('-' * 50)
        
        total = len(selected_labels)
        for label, count in zip(unique_labels, counts):
            percentage = (count / total) * 100
            class_name = class_names[label] if label < len(class_names) else str(label)
            print(f'{class_name:<10} {label:<10} {count:<10} {percentage:>6.2f}%')
        
        print('-' * 50)
        print(f'{"Total":<10} {"":<10} {total:<10} {"100.00%":<10}')
        
        # Check if balanced
        if len(unique_labels) == num_classes:
            expected_per_class = total / num_classes
            max_diff = np.max(np.abs(counts - expected_per_class))
            if max_diff <= 1:  # Allow for rounding differences
                print(f'\n[OK] Coreset appears to be balanced (expected ~{expected_per_class:.1f} per class)')
            else:
                print(f'\n[WARNING] Coreset is not perfectly balanced')
                print(f'  Expected ~{expected_per_class:.1f} per class, max difference: {max_diff:.1f}')
        
    except Exception as e:
        print(f'Error loading dataset: {e}')
        print('Could not compute class distribution.')

