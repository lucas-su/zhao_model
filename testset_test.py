from multiprocessing import Pool
import iitaff, os
import numpy as np

def dtest_testdataset(j):
    global i
    if np.isin(i, valid_dataset.__getitem__(j)['mask']):
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

    assert train_dataset.__getitem__(0)['mask'].shape[0] == 1 # turn off to_cat in data file

    for i in range(10):
        with Pool(16) as p:
            result = p.map(dtest_testdataset, list(range(valid_dataset.__len__())))
        print(f"key {i} has count {sum(result)}")

# test
# key 0 has count 2651
# key 1 has count 1138
# key 2 has count 348
# key 3 has count 551
# key 4 has count 340
# key 5 has count 1615
# key 6 has count 352
# key 7 has count 420
# key 8 has count 353
# key 9 has count 572

# train
# key 0 has count 4417
# key 1 has count 1841
# key 2 has count 582
# key 3 has count 831
# key 4 has count 590
# key 5 has count 2767
# key 6 has count 627
# key 7 has count 690
# key 8 has count 653
# key 9 has count 944

# val
# key 0 has count 1767
# key 1 has count 735
# key 2 has count 274
# key 3 has count 313
# key 4 has count 234
# key 5 has count 1098
# key 6 has count 251
# key 7 has count 289
# key 8 has count 270
# key 9 has count 367