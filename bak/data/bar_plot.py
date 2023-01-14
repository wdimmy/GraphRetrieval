import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

# set height of bar
IT = [0.53, 0.50, 0.51, 0.22, 0.11]
ECE = [0.53, 0.52, 0.53, 0.31, 0.22]

# Set position of bar on X axis
br1 = np.arange(len(IT))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, IT, color ='r', width = barWidth,
		edgecolor ='grey', label ='GCN')
plt.bar(br2, ECE, color ='g', width = barWidth,
		edgecolor ='grey', label ='Retrieval-enhanced GCN')

# Adding Xticks
plt.xlabel('Number of Per-class Samples in Training Dataset', fontweight ='bold', fontsize = 18)
plt.ylabel('Average Accuracy (%)', fontweight ='bold', fontsize = 18)
plt.xticks([r + barWidth/2 for r in range(len(IT))],
		['>5000\n(18 classes)', '1000--5000\n(43 classes)', '500--1000\n(49 classes)', '100-500\n(299 classes)', '<100\n(479 classes)'], fontsize = 18)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8],
		['0', '20', '40', '60', '80'], fontsize = 18)
plt.grid()
plt.legend(fontsize=18)
plt.savefig("upsto.png", bbox_inches='tight')
plt.show()
