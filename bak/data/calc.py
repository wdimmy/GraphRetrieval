import pandas as pd
#from tdc.utils import get_label_map
from collections import Counter
#from tdc.multi_pred import DDI
import matplotlib.pyplot as plt
from collections import defaultdict

# data = DDI(name = 'TWOSIDES')
# split = data.get_split()
from tdc.multi_pred import Catalyst


data = Catalyst(name = 'USPTO_Catalyst')
split = data.get_split()

train = split["train"]
valid = split["valid"]
test = split["test"]

# train["Y"].plot(kind="hist")
# plt.savefig('catalyst_train.png')

# test["Y"].plot(kind="hist")
# plt.savefig('catalyst_test.png')


train_labels = Counter([item for item in train["Y"]])
test_labels = Counter([item for item in test["Y"]])

print("Train total:", sum(train_labels.values()))
print("Test total:",  sum(test_labels.values()))

print(train_labels.most_common(10))
print(test_labels.most_common(10))

total_value = sum(train_labels.values())
results = [val for val in train_labels.values()]
results.sort(reverse=True)
fivethounsand = []
onethounsand  = []
fivehundred = []
onehundred = []
lesshundred = []

for item in train_labels.values():
    if item > 5000:
        fivethounsand.append(item)
    elif item > 1000:
        onethounsand.append(item)
    elif item > 500:
        fivehundred.append(item)
    elif item > 100:
        onehundred.append(item)
    else:
        lesshundred.append(item)

print(len(fivethounsand))
print(len(onethounsand))
print(len(fivehundred))
print(len(onehundred))
print(len(lesshundred))

print(results[:50])
print(results[-30:])
print("num of class", len(results))
#from tdc.utils import get_label_map
#get_label_map(name = 'USPTO_Catalyst', task = 'Catalyst')

# train_labels = defaultdict(int)
# for item in train["Y"]:
#     train_labels[item] += 1
#
#
# test_labels = defaultdict(int)
# for item in test["Y"]:
#     test_labels[item] += 1
#

# df = pd.read_csv("./uspto_catalyst.csv")

# labels = Counter([item for item in df["Y"]])
# total = sum(labels.values())
# print("total_number=", total)
# most_commons = labels.most_common(20)
# for item in most_commons:
#     print(item[0], item[1], "%.3f" % (item[1] / total))
#
# plt.bar(labels.keys(), labels.values())
# plt.ylim(0, 41000)
# plt.savefig('result.png')
#

#
# df["Y"].plot(kind="hist")
# plt.savefig('catalyst.png')