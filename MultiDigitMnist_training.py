# create test data----------------------------


def create_validation_data():
    # create directory for validation data : "MnistX_val", with 200 images of X digit numbers:
    number_of_cifers = 4
    val_target_directory = "Mnist" + str(number_of_cifers) + "_val"
    mumber_of_val_records = 200
    write_multi_records(val_target_directory, number_of_cifers, mumber_of_val_records)


def create_training_data():
    # creat directory for training data : "MnistX_val", with 1000 images of X digit numbers:
    number_of_cifers = 4
    train_target_directory = "Mnist" + str(number_of_cifers) + "_train"
    number_of_train_records = 1000
    write_multi_records(
        train_target_directory, number_of_cifers, number_of_train_records
    )


def train_split_model():
    from torch.utils.data import DataLoader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {device}")

    BATCH_SIZE = 32
    NOF_DIGITS = 4
    TRAIN_DIR = "Mnist" + str(NOF_DIGITS) + "_train"
    VAL_DIR = "Mnist" + str(NOF_DIGITS) + "_val"

    TRAIN_DS = MultiDigitMNISTDataset(source_dir=TRAIN_DIR)
    TRAIN_DL = DataLoader(TRAIN_DS, batch_size=BATCH_SIZE)

    VAL_DS = MultiDigitMNISTDataset(source_dir=VAL_DIR)
    VAL_DL = DataLoader(VAL_DS, batch_size=BATCH_SIZE)

    # start each training with a new model:
    split_model = None
    split_model = MultiDigitMNISTNet(nof_digits=NOF_DIGITS).to(device=device)

    # using weight_decay in SGD optimizer invoces L2 regularization
    #  - seems to be necessary since our wheigths and losses are extraordingly high with SGD- optimizer:
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5) # this optimizer is divergent most of the time leading to nan-size loss!

    optimizer = torch.optim.Adam(split_model.parameters())
    loss_fn = torch.nn.MSELoss()

    training(
        epochs=30,
        train_loader=TRAIN_DL,
        model=split_model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        show_progress=False,
        L2_regularization=True,
        L1_regularization=True,
        L1_lambda=0.01,
    )
    print("Training finished")


def validate_split_model():
    validate(model=model, train_loader=TRAIN_DL, val_loader=VAL_DL, loss_fn=loss_fn)

    print("Validation finished.")


def train_single_mnist_model():
    # training:
    from torch.utils.data import DataLoader

    # load datasets and create dataloader for BATCH_SIZE
    BATCH_SIZE = 64
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    train_dl = DataLoader(training_data, batch_size=BATCH_SIZE)

    # start with a new model each time:
    # model = None
    model = SingleDigitMNISTNet()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-5) # this optimizer is divergent most of the time leading to nan-size loss!
    optimizer = torch.optim.Adam(model.parameters())
    # our model outputs log_softmax(), i.e. we can use NLLLoss() here:
    loss_fn = torch.nn.NLLLoss()
    training(
        epochs=20,
        train_loader=train_dl,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        show_progress=True,
        L2_regularization=False,
        L1_regularization=False,
    )
    print("Training finished")


def validate_single_mnist_model():
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )
    test_dl = DataLoader(test_data, batch_size=BATCH_SIZE)
    validate(model=model, train_loader=train_dl, val_loader=test_dl, loss_fn=loss_fn)
    print("Validation finished.")
