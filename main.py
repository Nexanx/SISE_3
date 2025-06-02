from tabnet import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse



def main(): 
    parser = argparse.ArgumentParser(description="SISE_3")
    parser.add_argument("--load", type=str, help="Czy załadować z pliku")
    parser.add_argument("--new", action="store_true", help="Stwórz nową sieć")
    parser.add_argument("--save", type=str,help="Czy zapisać sieć")
    args = parser.parse_args()
    if args.load and args.new:
        parser.error("Można albo stworzyć nową albo załadowac z pliku sieć")
    
    X, y, cat_idxs, cat_dims = TabNetModel.load_secondary_mushroom()

    # 2) split 60/40:
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # 3) split holdout 50/50 → walidacja/test (czyli 20%/20% całości):
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_holdout, y_holdout,
        test_size=0.5,
        random_state=42,
        stratify=y_holdout
    )

    if args.new:
        model = TabNetModel(
            input_dim = X_train.shape[1],   
            output_dim = 2,                   
            cat_idxs = cat_idxs,
            cat_dims = cat_dims,
            n_d = 512,
            n_a = 512,
            n_steps = 3,
            gamma = 1.3,
            lambda_sparse = 1e-3,
            mask_type = "entmax",
            cat_emb_dim = 1,
            n_independent = 2,
            n_shared = 2,
            epsilon = 1e-15,
            seed = 1,
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
            X_valid = X_valid,
            y_valid = y_valid,
            max_epochs = 150,
            batch_size = 1024,
            virtual_batch_size = 512,
            loss_fn = torch.nn.CrossEntropyLoss(),
            eval_metric = ["accuracy"],
            patience = 50,
            weights = 1
        )
    elif args.load:
        model = TabNetModel.load(args.load)

    # Po treningu:
    preds = model.predict(X_test)
    
    
  
    history_df = pd.DataFrame.from_dict(model.model.history.history)


    plt.figure(figsize=(10, 5))
    plt.plot(history_df['loss'], label='Train Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Wykres accuracy
    if 'valid_accuracy' in history_df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(history_df['valid_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    cm = confusion_matrix(y_test, preds)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=['edible', 'poisonous'], yticklabels=['edible', 'poisonous'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_test, preds, digits=4, target_names=['edible', 'poisonous']))

    if args.save:
        model.save(args.save)

if __name__ == "__main__":
    main()