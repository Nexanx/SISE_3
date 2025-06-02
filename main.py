from tabnet import *
from sklearn.model_selection import train_test_split


X, y, cat_idxs, cat_dims = TabNetModel.load_secondary_mushroom()

print("Wymiary X:", X.shape)      
print("Wymiary y:", y.shape)       
print("cat_idxs:", cat_idxs)      
print("cat_dims:", cat_dims)       

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("X_train:", X_train.shape, "X_test:", X_test.shape)

model = TabNetModel(
    input_dim = X_train.shape[1],   
    output_dim = 2,                   
    cat_idxs = cat_idxs,
    cat_dims = cat_dims,
    n_d = 8,
    n_a = 8,
    n_steps = 3,
    gamma = 1.3,
    lambda_sparse = 1e-3,
    mask_type = "entmax",
    cat_emb_dim = 1,
    n_independent = 2,
    n_shared = 2,
    epsilon = 1e-15,
    seed = 0,
    momentum = 0.02,
    optimizer_fn = torch.optim.Adam,
    optimizer_params = {"lr": 2e-2, "weight_decay": 1e-4},
    scheduler_fn = torch.optim.lr_scheduler.StepLR,
    scheduler_params = {"step_size": 5, "gamma": 0.9},
    verbose = 1,
    device_name = "auto",
    n_shared_decoder = 1,
    n_indep_decoder = 1
)

# Trening z walidacją:
model.train_model(
    X_train = X_train,
    y_train = y_train,
    X_valid = X_test,
    y_valid = y_test,
    max_epochs = 5,
    batch_size = 512,
    virtual_batch_size = 64,
    loss_fn = nn.CrossEntropyLoss(),
    eval_metric = ["accuracy"],
    patience = 10,
    weights = 0,
    drop_last = False,
    num_workers = 0,
    callbacks = None
)

# Po treningu:
preds = model.predict(X_test)
test_acc = (preds == y_test).mean()
print("Test accuracy:", test_acc)
