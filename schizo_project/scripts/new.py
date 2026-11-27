import numpy as np

Xb = np.load("data/synthetic_signal_targeted/X_brain.npy")
sample = Xb[0]   # replace 0 with desired patient index
np.save("data/patient_brain_vector.npy", sample)
print("Saved single-sample brain vector to data/patient_brain_vector.npy")