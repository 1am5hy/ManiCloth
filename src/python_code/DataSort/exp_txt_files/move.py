import numpy as np

data = np.load("table_inference_left.npy", allow_pickle=True)[:, :3]
print(data.shape)

init = data[0]
data = data - init
