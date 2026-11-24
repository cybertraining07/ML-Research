
import os
import csv
import matplotlib.pyplot as plt

folder = "full_defense_evaluation"

csv_path = os.path.join(
    folder, "cifar10_full_defense_results.csv"
)

with open(csv_path, newline="") as file:
    results = list(csv.DictReader(file))

models = [r["Model"] for r in results]

clean = [float(r["Clean Accuracy (%)"]) for r in results]
fgsm = [float(r["FGSM Accuracy (%)"]) for r in results]
pgd = [float(r["PGD Accuracy (%)"]) for r in results]

# Generate comparison chart
x = range(len(models))
width = 0.25

plt.figure(figsize=(12, 7))

plt.bar(
    [i - width for i in x],
    clean,
    width,
    label="Clean"
)

plt.bar(
    list(x),
    fgsm,
    width,
    label="FGSM"
)

plt.bar(
    [i + width for i in x],
    pgd,
    width,
    label="PGD"
)

plt.xticks(list(x), models, rotation=15)
plt.ylabel("Accuracy (%)")
plt.title("CIFAR-10 Adversarial Defense Comparison")
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()

chart_path = os.path.join(
    folder, "cifar10_full_defense_comparison.png"
)

plt.savefig(chart_path, dpi=300)
plt.close()

# Regenerate report
report_path = os.path.join(
    folder, "cifar10_full_defense_report.txt"
)

with open(report_path, "w") as file:

    file.write("CIFAR-10 ADVERSARIAL DEFENSE EVALUATION\n")
    file.write("=" * 55 + "\n\n")

    file.write("Evaluation methodology:\n")
    file.write("Clean evaluation: 10,000 test images\n")
    file.write("FGSM evaluation: 10,000 test images\n")
    file.write("PGD evaluation: 2,000 balanced test images\n")
    file.write("FGSM epsilon: 8/255\n")
    file.write("PGD epsilon: 8/255\n")
    file.write("PGD step size: 2/255\n")
    file.write("PGD iterations: 10\n\n")

    file.write("Corrected evaluation results:\n\n")

    for result in results:
        for key, value in result.items():
            file.write(f"{key}: {value}\n")
        file.write("\n")

    file.write(
        "Research findings:\n"
        "- FGSM Defense achieved the highest adversarial "
        "accuracy against both tested attacks.\n"
        "- Mixed FGSM Defense preserved more clean accuracy "
        "than the adversarial-only defenses.\n"
        "- The baseline CNN was highly vulnerable to the "
        "tested adversarial attacks.\n\n"
    )

    file.write(
        "Limitations:\n"
        "- PGD results use a balanced 2,000-image subset.\n"
        "- Clean and FGSM results use all 10,000 test images.\n"
        "- Results apply only to the tested attacks and settings.\n"
        "- Empirical robustness does not establish universal robustness.\n"
        "- Each model was evaluated from one training run.\n"
    )

print("Research chart and report regenerated successfully.")
print(chart_path)
print(report_path)
