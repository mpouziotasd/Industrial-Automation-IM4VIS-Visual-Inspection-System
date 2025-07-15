import json
import matplotlib.pyplot as plt
import numpy as np

with open('results/results.json', 'r') as f:
    data_src = json.load(f)
print(data_src)


# Summary Statistics Init:
num_imgs = len(data_src.keys())
num_imgs_defective = 0
num_imgs_nondefective = 0
num_defects = 0


defect_areas = []
defect_roughness = []

for img_src in data_src.keys():
    if isinstance(data_src[img_src], dict):
        num_imgs_defective += 1
        num_defects = data_src[img_src]['num_defects']
        defect_areas.append(data_src[img_src]['defect_area'])
        defect_roughness.append(data_src[img_src]['defect_roughness'])
    else:
        num_imgs_nondefective += 1 


x = [num_imgs_nondefective, num_imgs_defective]
labels = ['Non Defective', 'Defective']
colors = ['#2840ad', "#a12a2a"]

plt.bar(labels, x, color=colors)
plt.ylabel('Number of Images')
plt.title('Defective vs Non-Defective Images')
plt.show()

avg_area = np.mean(defect_areas)
avg_roughness = np.mean(defect_roughness)
print(f"Average Defect Area: {avg_area:.2f}")
print(f"Average Roughness: {avg_roughness:.2f}")
plt.scatter(defect_areas, defect_roughness, color='red', alpha=0.6)
plt.xlabel('Defect Area')
plt.ylabel('Defect Roughness')
plt.title('Defect Area vs Roughness')
plt.grid(True)
plt.show()
