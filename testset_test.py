from multiprocessing import Pool
import iitaff, os
import numpy as np

def dtest_testdataset(j):
    global i
    if np.isin(i, valid_dataset.__getitem__(j)['mask'].flatten()):
        return 1
    else:
        return 0

if __name__ == "__main__":
    if os.path.exists("devmode"):
        root = "/media/luc/data/iitaff"
    else:
        root = "/home/schootuiterkampl/iitaff"

    train_dataset = iitaff.iitaff(root, "train")
    valid_dataset = iitaff.iitaff(root, "valid")
    test_dataset = iitaff.iitaff(root, "test")
    n_cpu = os.cpu_count()


    for i in range(10):
        with Pool(16) as p:
            result = p.map(dtest_testdataset, list(range(valid_dataset.__len__())))
        print(f"key {i} has count {sum(result)}")



# test

# key 0 has count 2651
# key 1 has count 2651
# key 2 has count 0
# key 3 has count 0
# key 4 has count 0
# key 5 has count 0
# key 6 has count 0
# key 7 has count 0
# key 8 has count 0
# key 9 has count 0

# train
# key 0 has count 4417
# key 1 has count 4417
# key 2 has count 0
# key 3 has count 0
# key 4 has count 0
# key 5 has count 0
# key 6 has count 0
# key 7 has count 0
# key 8 has count 0
# key 9 has count 0

# val
# key 0 has count 1767
# key 1 has count 1767
# key 2 has count 0
# key 3 has count 0
# key 4 has count 0
# key 5 has count 0
# key 6 has count 0
# key 7 has count 0
# key 8 has count 0
# key 9 has count 0