import numpy as np

array = np.array([0,20,10,30])
indices = np.linspace(0, 1, 4)

loss = 0
alpha = 0.1
natural_indices = np.argsort(indices)
current_array = array[natural_indices]

for i in natural_indices:
    if i == 0:
        continue

    left_item = current_array[i-1]
    this_item = current_array[i]

    loss_left = left_item - this_item
    if loss_left <= 0:
        continue
    
    print(f"loss_left: {loss_left}, i: {i}")
    loss = loss + (loss_left * alpha)

print(f"loss: {loss}")

     
