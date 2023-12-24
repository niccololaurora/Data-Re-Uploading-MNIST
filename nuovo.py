from qibo import Circuit, gates
import numpy as np
import imageio
import os
from qiskit.visualization import plot_state_qsphere
import matplotlib.pyplot as plt
from help_functions import visualize_state_sequence, states_visualization


c = Circuit(8)
c.add(gates.RX(0, theta=0.4))
c.add(gates.RY(1, theta=0.4))
c.add(gates.RZ(2, theta=0.4))
c.add(gates.RX(3, theta=0.4))
c.add(gates.RY(4, theta=0.4))
c.add(gates.RZ(5, theta=0.4))
c.add(gates.RX(6, theta=0.4))
c.add(gates.RY(7, theta=0.4))

"""
c = Circuit(8)
c.add(gates.RX(0, theta=0.4))
c.add(gates.RY(1, theta=0.4))
c.add(gates.RZ(2, theta=0.4))
c.add(gates.RX(3, theta=0.4))
c.add(gates.RY(4, theta=0.4))
c.add(gates.RZ(5, theta=0.4))
c.add(gates.RX(6, theta=0.4))
c.add(gates.RY(7, theta=0.4))


stato = c().state(numpy=True)

fig, ax = plt.subplots(figsize=(10, 10))
plot_state_qsphere(stato, ax=ax, show_state_labels=False)
fig.text(0.2, 0.8, "Iteration 1", fontsize=20)
fig.savefig("prova.png")
"""

stato = []
for i in range(10):
    param = np.random.rand(8)
    c.set_parameters(param)
    s = c().state(numpy=True)
    stato.append(s)


# Esempio di utilizzo
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

# Genera le immagini
for epoch in range(10):
    states_visualization(stato[epoch], f"{output_folder}/image_{epoch}.png", epoch)

# Crea la GIF animata
image_files = [f"{output_folder}/image_{epoch}.png" for epoch in range(10)]
output_gif = "animated_qsphere.gif"
images = [imageio.imread(file) for file in image_files]
imageio.mimsave(output_gif, images, duration=600)

print(f"Animated GIF saved as {output_gif}")
