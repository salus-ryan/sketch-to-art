"""Generate training curve figure for BrailleNet paper section."""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

with open('../braillenet_n8_final_results.json') as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle('BrailleNet Foundation Model — Training Curves (H100, n=8)',
             fontsize=13, fontweight='bold', y=0.98)

colors = {
    's1': '#4CAF50',
    's2': '#2196F3',
    's3a': '#FF9800',
    's3b': '#9C27B0',
    's35': '#F44336',
    's4': '#00BCD4',
}

# Stage 1: Perception
ax = axes[0, 0]
curve = data['s1_perception']['curve']
epochs = [r['epoch'] for r in curve]
ax.plot(epochs, [r['eval_cell'] for r in curve], color=colors['s1'], linewidth=2)
ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='target (95%)')
ax.set_title('Stage 1: Stroke → Cell', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cell Accuracy')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=8)
ax.text(80, data['s1_perception']['best_cell_acc'] + 0.02,
        f"{data['s1_perception']['best_cell_acc']:.1%}", color=colors['s1'], fontsize=9)

# Stage 2: Generation
ax = axes[0, 1]
curve = data['s2_generation']['curve']
epochs = [r['epoch'] for r in curve]
ax.plot(epochs, [r['pen_acc'] for r in curve], color=colors['s2'], linewidth=2)
ax.set_title('Stage 2: Cell → Stroke', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Pen State Accuracy')
ax.set_ylim(0, 1.05)
ax.text(80, 0.92, f"{data['s2_generation']['best_pen_acc']:.1%}",
        color=colors['s2'], fontsize=9)

# Stage 3a: Text
ax = axes[0, 2]
curve = data['s3a_text']['curve']
epochs = [r['epoch'] for r in curve]
ax.plot(epochs, [r['acc'] for r in curve], color=colors['s3a'], linewidth=2)
ax.axhline(0.95, color='gray', linestyle='--', alpha=0.5, label='target (95%)')
ax.set_title('Stage 3a: Cell → Text', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Char Accuracy')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=8)

# Stage 3b: Sequence
ax = axes[1, 0]
curve = data['s3b_sequence']['curve']
epochs = [r['epoch'] for r in curve]
ax.plot(epochs, [r['cell_acc'] for r in curve], color=colors['s3b'], linewidth=2)
ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5, label='target (90%)')
ax.set_title('Stage 3b: Sequence Composition', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Cell Accuracy')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=8)

# Stage 3.5: Algebra
ax = axes[1, 1]
curve = data['s35_algebra']['curve']
epochs = [r['epoch'] for r in curve]
ax.plot(epochs, [r['vca'] for r in curve], color=colors['s35'], linewidth=2, label='vector')
ax.plot(epochs, [r['sa'] for r in curve], color=colors['s35'], linewidth=2,
        linestyle='--', alpha=0.7, label='scalar')
ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Stage 3.5: Ternary Algebra', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=8)

# Stage 4: End-to-end
ax = axes[1, 2]
curve = data['s4_e2e']['curve']
epochs = [r['epoch'] for r in curve]
ax.plot(epochs, [r['text_acc'] for r in curve], color=colors['s4'], linewidth=2)
ax.axhline(0.90, color='gray', linestyle='--', alpha=0.5, label='target (90%)')
ax.set_title('Stage 4: End-to-End (Stroke → Text)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Text Accuracy')
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('figures/braillenet_training_curves.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/braillenet_training_curves.png', bbox_inches='tight', dpi=300)
print("Saved: figures/braillenet_training_curves.{pdf,png}")
