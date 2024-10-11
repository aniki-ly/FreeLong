import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d

def interpolate_array(input_array, output_length):
    original_indices = np.linspace(0, len(input_array) - 1, num=len(input_array))
    new_indices = np.linspace(0, len(input_array) - 1, num=output_length)
    interpolation_function = interp1d(original_indices, input_array, kind='linear')
    interpolated_array = interpolation_function(new_indices)
    return interpolated_array

def interpolate_focus(array, central_index, scale):
    if scale <= 0 or not isinstance(scale, int):
        raise ValueError("Scale must be a positive integer.")
    if not (0 <= central_index < len(array)):
        raise ValueError("Central index must be within the bounds of the array.")
    
    final_array = np.zeros(len(array))
    left_part = array[:central_index + 1]
    right_part = array[central_index:]

    scale_left = max(1, len(left_part) // scale)
    scale_right = max(1, len(right_part) // scale)

    interpolated_left = interpolate_array(left_part, scale_left)
    interpolated_right = interpolate_array(right_part, scale_right)

    # Correct indexing for insertion
    final_array[central_index - len(interpolated_left) + 1: central_index + 1] = interpolated_left
    final_array[central_index + 1: central_index + 1 + len(interpolated_right)] = interpolated_right

    return final_array


# Create an empty 16x16 matrix
attention_matrix_16 = np.zeros((16, 16))

# Set the main diagonal to 1 (the highest attention)
np.fill_diagonal(attention_matrix_16, 1)

# Gaussian decay parameters
sigma = 1  # Control the spread of the Gaussian function

# Fill the matrix with Gaussian decay values around the diagonal
for i in range(16):
    for offset in range(1, 3):  # Consider two nearest neighbors
        if i - offset >= 0:
            attention_matrix_16[i, i - offset] = np.exp(-(offset**2) / (2 * sigma**2))
        if i + offset < 16:
            attention_matrix_16[i, i + offset] = np.exp(-(offset**2) / (2 * sigma**2))


# Perform bicubic interpolation
attention_matrix_64 = zoom(attention_matrix_16, (4, 4), order=3)  # order=3 indicates bicubic interpolation

scale = 64 // 16
rescaled_matrix = np.zeros((64, 64))
for idx in range(64):
    rescaled_matrix[idx] = interpolate_focus(attention_matrix_64[idx], idx, scale)

plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix_64, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Interpolated 64x64 Attention Matrix Using Bicubic Interpolation')
# save
plt.savefig('attention_matrix_64.png')

# Visualize the attention matrix using matplotlib
plt.figure(figsize=(8, 6))
plt.imshow(attention_matrix_16, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Gaussian Decay Attention Matrix')
# save 
plt.savefig('attention_matrix_16.png')


# Visualize the recalsed attention matrix using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(rescaled_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Rescaled Attention Matrix from 64x64 to 16x64')
# save
plt.savefig('rescaled_matrix_64_to_16.png')
