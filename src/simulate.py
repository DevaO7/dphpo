from flearn.trainmodel import models
from flearn.servers.server_avg import FedAvg
from utils.data_utils import get_data_loaders, visualize_partition
import torch
import numpy as np
import os
import random
import csv
import matplotlib.pyplot as plt

def set_seed(seed=42):
    # 1. Python & NumPy
    random.seed(seed)
    np.random.seed(seed)
    
    # 2. PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 3. PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
    
    # 4. Force Deterministic Algorithms
    # Warning: This can slow down training slightly
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_optimum(cfg, train_loader, test_loader, epochs=1000, log_interval=50, lr=0.01):
    # Load the model
    if cfg.dataset.name == 'synthetic':
        model = getattr(models, cfg.dataset.model_name)(input_dim=cfg.dataset.dim_input, output_dim=cfg.dataset.dim_output)
    elif cfg.dataset.name == 'mnist':
        model = getattr(models, cfg.dataset.model_name)()
    device = torch.device(cfg.run_settings.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)
    loss_fn = getattr(torch.nn, cfg.run_settings.loss_function)()
    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % log_interval == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for test_X, test_y in test_loader:
                    test_X, test_y = test_X.to(device), test_y.to(device)
                    outputs = model(test_X)
                    predicted = outputs.argmax(dim=1)
                    total += test_y.size(0)
                    correct += (predicted == test_y).sum().item()
            accuracy = 100 * correct / total
            print(f"Epoch {epoch}, Test Accuracy: {accuracy:.2f}%")


def compile_tuning_results(cfg):
    if cfg.dataset.name == 'synthetic':
        similarity = (cfg.dataset.alpha, cfg.dataset.beta)
    else:
        similarity = cfg.dataset.similarity
    client_ratios = {}
    evaluation_metrics = ['test_accuracy', 'test_loss', 'train_accuracy', 'train_loss']
    for client_ratio in cfg.results.client_ratios:
        if cfg.tuning.cross_validation:
            results = {}
            best_hyperparameter = None
            best_accuracy = 0.0
            for hyperparameter in cfg.tuning.hyperparameter_grid:
                results[hyperparameter] = {}
                save_path = os.path.join(cfg.tuning.save_path, str(similarity), f"{client_ratio}ur", f"{hyperparameter}beta")
                best_test_accuracy = []
                best_train_accuracy = []
                for fold in range(cfg.tuning.cv_folds):
                    file_name = f"fold_{fold}"
                    with open(os.path.join(save_path, f"{file_name}.csv"), mode='r') as file:
                        reader = csv.reader(file)
                        next(reader)  # Skip header row
                        test_accuracies = []
                        train_accuracies = []
                        train_losses = []
                        test_losses = []
                        for row in reader:
                            test_accuracies.append(float(row[4]))  # Test Accuracy is the 5th column
                            train_accuracies.append(float(row[3]))  # Train Accuracy is the 4th column
                            train_losses.append(float(row[1]))      # Train Loss is the 2nd column
                            test_losses.append(float(row[2]))       # Test Loss is the 3rd column
                        best_test_accuracy.append(max(test_accuracies))
                        best_train_accuracy.append(max(train_accuracies))
                avg_best_test_accuracy = sum(best_test_accuracy) / len(best_test_accuracy)
                avg_best_train_accuracy = sum(best_train_accuracy) / len(best_train_accuracy)
                std_train_accuracy = np.std(best_train_accuracy)
                std_test_accuracy = np.std(best_test_accuracy)
                if avg_best_train_accuracy > best_accuracy:
                    best_accuracy = avg_best_train_accuracy
                    best_hyperparameter = hyperparameter
                results[hyperparameter]['test_accuracy'] = test_accuracies
                results[hyperparameter]['train_accuracy'] = train_accuracies
                results[hyperparameter]['train_loss'] = train_losses
                results[hyperparameter]['test_loss'] = test_losses
                results[hyperparameter]['avg_best_test_accuracy'] = round(avg_best_test_accuracy, 4)
                results[hyperparameter]['avg_best_train_accuracy'] = round(avg_best_train_accuracy, 4)
                results[hyperparameter]['std_train_accuracy'] = round(std_train_accuracy, 4)
                results[hyperparameter]['std_test_accuracy'] = round(std_test_accuracy, 4)
                client_ratios[client_ratio] = results

            print('Writing tuning summary to file and plotting results...')
            # Plot and write it to a file
            with open(os.path.join(cfg.tuning.save_path, str(similarity), f"{client_ratio}ur", "tuning_summary.txt"), mode='w', newline='') as file:
                for hyperparameter, metrics in results.items():

                    file.write(f"Hyperparameter: {hyperparameter}\n")
                    file.write(f"Avg Best Train Accuracy: {metrics['avg_best_train_accuracy']}\n")
                    file.write(f"Std Train Accuracy: {metrics['std_train_accuracy']}\n")
                    file.write(f"Avg Best Test Accuracy: {metrics['avg_best_test_accuracy']}\n")
                    file.write(f"Std Test Accuracy: {metrics['std_test_accuracy']}\n")
                    file.write("\n")
                file.write(f"Best Hyperparameter: {best_hyperparameter} with Avg Best Train Accuracy: {round(best_accuracy, 4)}\n")
            for metric in evaluation_metrics:
                plt.figure()
                for hyperparameter in cfg.tuning.hyperparameter_grid:
                    if metric == 'train_accuracy':
                        plt.plot(results[hyperparameter]['train_accuracy'], label=f"Beta: {hyperparameter}")
                    elif metric == 'test_accuracy':
                        plt.plot(results[hyperparameter]['test_accuracy'], label=f"Beta: {hyperparameter}")
                    elif metric == 'train_loss':
                        plt.plot(results[hyperparameter]['train_loss'], label=f"Beta: {hyperparameter}")
                    elif metric == 'test_loss':
                        plt.plot(results[hyperparameter]['test_loss'], label=f"Beta: {hyperparameter}")
                plt.xlabel("Communication Round")
                plt.ylabel(metric)
                plt.title(f"{metric} vs Communication Rounds")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.tuning.save_path, str(similarity), f"{client_ratio}ur", f"{metric.lower().replace(' ', '_')}_comparison.png"))
                plt.close()
    if cfg.tuning.cross_validation and cfg.results.client_ratio_vs_beta:
        for metric in evaluation_metrics:
            for hyperparameter in cfg.tuning.hyperparameter_grid:
                plt.figure()
                for client_ratio in cfg.results.client_ratios:
                    plt.plot(client_ratios[client_ratio][hyperparameter][metric], label=f"Client Ratio: {client_ratio}")
                plt.xlabel("Communication Round")
                plt.ylabel(metric)
                plt.title(f"{metric} vs Communication Rounds for Hyperparameter: {hyperparameter}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(cfg.tuning.save_path, str(similarity), f"client_ratio_comparison_{metric.lower().replace(' ', '_')}_hyperparameter_{hyperparameter}.png"))
                plt.close()
                    


def tune_hyperparameters(cfg):
    set_seed(cfg.run_settings.seed)
    if cfg.dataset.name == 'synthetic':
        model = getattr(models, cfg.dataset.model_name)(input_dim=cfg.dataset.dim_input, output_dim=cfg.dataset.dim_output)
    else:
        model = getattr(models, cfg.dataset.model_name)()

    loss_fn = getattr(torch.nn, cfg.dataset.loss_fn)(reduction='mean')
    if cfg.dataset.name == 'synthetic':
        similarity = (cfg.dataset.alpha, cfg.dataset.beta)
    else:
        similarity = cfg.dataset.similarity
    if cfg.tuning.cross_validation:
        for hyperparameter in cfg.tuning.hyperparameter_grid:
            print(f"Tuning hyperparameter: {hyperparameter}")
            if cfg.server.constant_global_step:
                global_step = cfg.server.global_step_size
                local_step = hyperparameter
            else:
                global_step = (cfg.server.client_ratio*cfg.dataset.nb_users)**0.5
                local_step = hyperparameter/(cfg.server.local_updates*global_step)
            
            #TODO: Save Path needs to be updated according to hyperparameter and fold
            save_path = os.path.join(cfg.tuning.save_path, str(similarity), f"{cfg.server.client_ratio}ur", f"{hyperparameter}beta")
            os.makedirs(save_path, exist_ok=True)

            for fold in range(cfg.tuning.cv_folds):
                #TODO Load appropriate Data Loaders according to fold
                train_data_loader, test_data_loader = get_data_loaders(cfg, per_client_loader=True)
                if cfg.run_settings.visualize_data_partition:
                    visualize_partition(cfg, train_data_loader, test_data_loader, save_path)
                    exit('Visualization done. Exiting now.')
                file_name = f"fold_{fold}"
                print(f"Starting cross-validation fold {fold + 1}/{cfg.tuning.cv_folds}")
                server = FedAvg(
                    model=model,
                    train_data_loader=train_data_loader,
                    test_data_loader=test_data_loader,
                    save_path=save_path,
                    file_name=file_name,
                    num_glob_iters=cfg.run_settings.rounds,
                    loss_fn_name=cfg.dataset.loss_fn, 
                    local_learning_rate=local_step,
                    global_learning_rate=global_step,
                    weight_decay=cfg.server.weight_decay,
                    use_cuda=cfg.run_settings.use_cuda, 
                    similarity=similarity, 
                    client_ratio=cfg.server.client_ratio, 
                    dp=cfg.server.dp, 
                    local_updates=cfg.server.local_updates, 
                    sample_rate=cfg.server.sampling_rate, 
                    noise_multiplier=cfg.server.sigma, 
                    max_grad_norm=cfg.server.max_grad_norm, 
                    x_label=cfg.dataset.x_label,
                    y_label=cfg.dataset.y_label, 
                    resume=cfg.run_settings.resume_from_checkpoint
                )
                server.train()
    else:
        pass

