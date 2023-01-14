import pickle
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


im = Image.open('./regression_gain.png')
height = im.size[1]
width = im.size[0]

with open("pcqm4m_train.pkl", "rb") as file:
    results = pickle.load(file)
    labels = Counter(results)

sorted_results = sorted(labels.items(), key=lambda item: item[0])
x_test = []
y_test = []

for key, value in sorted_results:
    x_test.append(key)
    y_test.append(value)


with open("pcqm4m_test.pkl", "rb") as file:
    results = pickle.load(file)
    labels = Counter(results)

sorted_results = sorted(labels.items(), key=lambda item: item[0])
x_train = []
y_train = []
for key, value in sorted_results:
    x_train.append(key)
    y_train.append(value)

fig, ax = plt.subplots()
plt.plot(x_test, y_test, "k")
plt.plot(x_train, y_train, "g")
plt.ylabel("Frequency", fontsize=15)
plt.xlabel("Numeric Value", fontsize=15)
ax.yaxis.set_tick_params(labelsize=15)
ax.xaxis.set_tick_params(labelsize=15)

plt.legend(["train data distribution", "test data distribution"], ncol=2, framealpha=0.0, fontsize=10)

sub_ax = plt.axes([.42, .29, .54, .58])

labels = ['[0-10)', '[10-20)', '[20,30)', '>=30']
men_means = [0.155, 0.183, 0.241, 0.209]
women_means = [0.151, 0.169, 0.199, 0.186]
gains = [round((item1-item2)/item1 *100, 1) for item1, item2 in zip(men_means, women_means)]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


rects1 = sub_ax.bar(x - width/2, men_means, width, label='GCN')
rects2 = sub_ax.bar(x + width/2, women_means, width, label='Retrieval-enhanced GCN')

# Add some text for labels, title and custom x-axis tick labels, etc.
sub_ax.set_ylabel('Average MAE', fontsize=12)
sub_ax.set_xlabel('Numeric Range', fontsize=12)
sub_ax.set_xticks(x, labels, fontsize=10)
sub_ax.yaxis.set_tick_params(labelsize=10)
sub_ax.legend(fontsize=8,framealpha=0.0)

#ax.bar_label(rects1, padding=3)
sub_ax.bar_label(rects2, labels=["  {}%\n".format(gains[i]) + u'\u2193' for i in range(4)], color="red", fontsize=12, padding=3)

# fig = plt.figure()
# fig.tight_layout()
# plt.savefig("regression_gain.png")
# plt.show()
plt.tight_layout()
plt.savefig("regression.png", dpi=199)
plt.show()

