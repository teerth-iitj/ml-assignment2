import json
with open("q3a_results.json") as f:
    s = json.load(f)["summary"]
print(f"{'model':>9} {'clean acc':>10} {'masked acc':>11} {'drop':>6} {'bias2':>7} {'variance':>9}")
rows = []
for k in ["tree", "bagging", "boosting"]:
    m = s[k]
    drop = m["acc_mean"] - m["masked_acc_mean"]
    rows.append({"model": k, "clean": m["acc_mean"], "masked": m["masked_acc_mean"],
                 "drop": drop, "bias2": m["bias2"], "variance": m["variance"]})
    print(f"{k:>9} {m['acc_mean']:>10.3f} {m['masked_acc_mean']:>11.3f} {drop:>6.3f} "
          f"{m['bias2']:>7.3f} {m['variance']:>9.4f}")
with open("q3d_table.json", "w") as f:
    json.dump(rows, f, indent=2)