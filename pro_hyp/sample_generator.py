import os
import numpy as np

np.random.seed(42)
def generate_b_samples(size, b_min, b_max):
    u = np.random.uniform(size=size)
    b_samples = b_max * b_min / (b_max - u * (b_max - b_min))
    return b_samples

def generate_e_samples(size, e_min, e_max):
    e_samples = np.random.uniform(e_min, e_max, size=size)
    return e_samples

num_samples = 1000
b_min = 70
b_max = 300
e_min = 1.05
e_max = 1.5

b_samples = generate_b_samples(num_samples, b_min, b_max)
e_samples = generate_e_samples(num_samples, e_min, e_max)

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
output_path = os.path.join(src_dir, 'samples.csv')
np.savetxt(output_path, np.column_stack((b_samples, e_samples)), delimiter=',', header='b_samples,e_samples', comments='')
