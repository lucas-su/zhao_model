from multiprocessing import Pool
import iitaff, os
import numpy as np
from collections import Counter

def dtest_testdataset(j):
    global c
    c.update(valid_dataset.__getitem__(j)['mask'].flatten())

if __name__ == "__main__":
    if os.path.exists("devmode"):
        root = "/media/luc/data/iitaff"
    else:
        root = "/home/schootuiterkampl/iitaff"

    train_dataset = iitaff.iitaff(root, "train")
    valid_dataset = iitaff.iitaff(root, "valid")
    test_dataset = iitaff.iitaff(root, "test")
    n_cpu = os.cpu_count()

    assert train_dataset.__getitem__(0)['mask'].shape[0] > 10 # turn off to_cat in data file

    c = Counter()

    # with Pool(16) as p:
    #     p.map(dtest_testdataset, list(range(valid_dataset.__len__())))
    ## Counter doesn't work well in parallel
    print(test_dataset.__len__())
    for j in list(range(test_dataset.__len__())):
        if j % 500 == 0:
            print(j)
        c.update(test_dataset.__getitem__(j)['mask'].flatten())
    print(c.items())

# label counts:

# train
# (0.0, 242845023)
# (1.0, 15382425)
# (2.0, 670580)
# (3.0, 9332078)
# (4.0, 1060225)
# (5.0, 7407467)
# (6.0, 1622429)
# (7.0, 1111025)
# (8.0, 1314111)
# (9.0, 8727149)

# val:
# (0.0, 97639810)
# (1.0, 6141107)
# (2.0, 269611)
# (3.0, 3678676)
# (4.0, 386657)
# (5.0, 2727832)
# (6.0, 574402)
# (7.0, 422488)
# (8.0, 500451)
# (9.0, 3461078)

# test
# (0.0, 145840755)
# (1.0, 9161842)
# (2.0, 326314)
# (3.0, 6368899)
# (4.0, 773987)
# (5.0, 4096654)
# (6.0, 707461)
# (7.0, 696079)
# (8.0, 634975)
# (9.0, 5128970)

# total:
# (0.0, 486325588)
# (1.0, 30685374)
# (2.0, 1266505)
# (3.0, 19379653)
# (4.0, 2220869)
# (5.0, 14231953)
# (6.0, 2904292)
# (7.0, 2229592)
# (8.0, 2449537)
# (9.0, 17317197)



# document counts:
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