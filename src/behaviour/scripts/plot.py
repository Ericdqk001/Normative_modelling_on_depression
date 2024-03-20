from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

syndrome_map = {
    "Anxiety/Depression": "cbcl_scr_syn_anxdep_t",
    "Withdrawal/Depression": "cbcl_scr_syn_withdep_t",
    "Somatic": "cbcl_scr_syn_somatic_t",
    "Social": "cbcl_scr_syn_social_t",
    "Thought": "cbcl_scr_syn_thought_t",
    "Attention": "cbcl_scr_syn_attention_t",
    "RuleBreak": "cbcl_scr_syn_rulebreak_t",
    "Aggressive": "cbcl_scr_syn_aggressive_t",
}

lca_prob_class_path = Path("src/behaviour/files/lcmodel_prob_class.csv")

lca_prob_class = pd.read_csv(lca_prob_class_path, index_col=0)


prob_pr2 = lca_prob_class[lca_prob_class["Var2"] == "Pr(2)"].copy()

inverse_syndrome_map = {v: k for k, v in syndrome_map.items()}

# Update the values in the 'L2' column to their keys in the syndrome_map
prob_pr2["L2"] = prob_pr2["L2"].map(inverse_syndrome_map)

prob_pr2.head()

prob_pr2["Class"] = prob_pr2["Var1"].str.extract(r"class (\d+):").astype(int)

pivot_pr2 = prob_pr2.pivot(index="L2", columns="Class", values="value")

# Swap the row order for better visualization
new_order = [
    "Aggressive",
    "RuleBreak",
    "Attention",
    "Thought",
    "Anxiety/Depression",
    "Withdrawal/Depression",
    "Somatic",
    "Social",
]

pivot_pr2 = pivot_pr2.reindex(new_order)

# Plot a line chart
plt.figure(figsize=(14, 10))
for class_col in pivot_pr2.columns:
    plt.plot(
        pivot_pr2.index, pivot_pr2[class_col], marker="o", label=f"Class {class_col}"
    )

plt.title("Conditional Probability of Each Variable Being 2 (pr(2)) by Class")
plt.xlabel("Variables")
plt.ylabel("Conditional Probability")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend()  # Add a legend
plt.tight_layout()

plt.show()
