import os
import sys
import numpy as np
import matplotlib.pyplot as plt

log_scale = False

# Allow for a version to be provided at the command line, as in
# $ python3 show_results.py v15
if len(sys.argv) > 1:
    version = sys.argv[1]
else:
    version = "test"

accuracy_results_path = f"results/accuracy_{version}.npy"
accuracy_history_path = f"results/accuracy_history_{version}.npy"
results_path = f"plots/results_{version}.png"
results_detail_path = f"plots/results_detail_{version}.png"
os.makedirs("plots", exist_ok=True)

fig = plt.figure("error_curves")
ax = fig.gca()
fig_detail = plt.figure("error_curves_detail")
ax_detail = fig_detail.gca()

accuracy_results = np.load(accuracy_results_path)
test_errors = 1 - accuracy_results
test_mean = np.mean(test_errors) * 100
test_stddev = np.var(test_errors)**.5 * 100
test_stderr = test_stddev / test_errors.size ** .5

print()
print(f"testing errors for version {version}")
print(f"mean  : {test_mean:.04}%")
print(f"stddev: {test_stddev:.04}%")
print(f"stderr: {test_stderr:.04}%")
print(f"n runs: {test_errors.size}")
print()

accuracy_histories = np.load(accuracy_history_path).transpose()
raw_errors = 1 - accuracy_histories
if log_scale:
    errors = np.log10(raw_errors)
else:
    errors = 100 * raw_errors

ax.plot(errors, linewidth=.5, alpha=.8)
ax_detail.plot(errors)

if log_scale:
    ax.set_ylabel("log10(error)")
else:
    ax.set_ylabel("Error (percent)")
ax.set_xlabel("Epoch")
ax.grid()

if log_scale:
    ax_detail.set_ylabel("log10(error)")
else:
    ax_detail.set_ylabel("Error (percent)")
ax_detail.set_xlabel("Epoch")

final_errors = errors[-1, :]
final_mean = np.mean(final_errors)
final_stddev = np.var(final_errors)**.5
ax_detail.set_ylim(
    final_mean - 3 * final_stddev,
    final_mean + 3 * final_stddev)
ax_detail.grid()

plt.figure("error_curves")
plt.savefig(results_path, dpi=300)
plt.figure("error_curves_detail")
plt.savefig(results_detail_path, dpi=300)
