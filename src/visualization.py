import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_distribution(y, title, save_path=None):
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=y, palette='viridis')
    plt.xlabel('Risk Level', fontsize=16, fontweight='bold')
    plt.ylabel('Count', fontsize=16, fontweight='bold')

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max([p.get_height() for p in ax.patches]) + 10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
