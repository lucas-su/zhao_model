def arg_int(args):
    if "dataset" in args.keys():
        dataset = args["dataset"] # options 'sunrgbd' 'iitaff' 'umd'
    else:
        print("using default dataset sunrgbd")
        dataset = 'sunrgbd'

    if "dcnn_type" in args.keys():
        dcnn = args["dcnn_type"] # options 'resnet50' 'resnet18' 'vgg'
    else:
        print("using default dcnn type resnet50")
        dcnn = "resnet50"

    if "train_test_mode" in args.keys():
        train_test_mode = args["train_test_mode"] # options 'train' 'test'
    else:
        print("training network by default")
        train_test_mode = 'train'

    if "train_dcnn" in args.keys():
        train_dcnn = True if args["train_dcnn"].lower() == 'true' else False # options true, false
    else:
        print("training dcnn layers by default")
        train_dcnn = True

    if "device" in args.keys():
        use_gpu = True if args["device"].lower() == 'gpu' else False  # options true, false
    else:
        print("training on GPU by default")
        use_gpu = True

    if "norm" in args.keys():
        norm = True if args["norm"].lower() == 'true' else False  # options true, false
    else:
        print("using batch norm by default")
        norm = True

    if "dropout" in args.keys():
        dropout = True if args["dropout"].lower() == 'true' else False
    else:
        print("using dropout by default")
        dropout = True

    if "activation" in args.keys():
        activation = True if args["activation"].lower() == 'true' else False
    else:
        print("using activation relu by default")
        activation = True
    return dataset, dcnn, train_test_mode, train_dcnn, use_gpu, norm, dropout, activation
