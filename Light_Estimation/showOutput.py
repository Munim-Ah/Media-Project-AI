from matplotlib import pyplot as plt

import multilum
import ProbeEval
import ProbeModel

# load images for three scenes, 2 light directions
# access images as I[scene_index, light_index]
I = multilum.query_images(
    scenes=['main_experiment120',
            'kingston_dining10',
            'elm_revis_kitchen14'],
    dirs=[14,24]
)

J = multilum.query_probes(
    scenes=['main_experiment120',
            'kingston_dining10',
            'elm_revis_kitchen14'],
    dirs=[14,24]
)

# visualize in 2x3 grid
for i in range(3):
    plt.subplot(2,3,i+1)
    plt.imshow(I[i,0])

    plt.subplot(2,3,i+4)
    plt.imshow(I[i,1])
plt.show()


for j in range(3):
    plt.subplot(2,3,j+1)
    plt.imshow(J[j,0])

    plt.subplot(2,3,j+4)
    plt.imshow(J[j,1])

plt.show()
