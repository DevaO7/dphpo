from flearn.trainmodel import models
from flearn.servers.server_avg import FedAvg
from utils.data_utils import get_data_loaders, visualize_partition
from utils.tuning_utils import perform_early_stopping_analysis, perform_simple_cross_validation_analysis
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
        if cfg.dataset.iid:
            similarity = 'iid'
        else:
            similarity = (cfg.dataset.alpha, cfg.dataset.beta)
    else:
        similarity = cfg.dataset.similarity
    client_ratios = {}
    evaluation_metrics = ['test_accuracy', 'test_loss', 'train_accuracy', 'train_loss']
    if cfg.tuning.parameter_to_tune == 'step_size':
        save_path = os.path.join(cfg.tuning.save_path+f'_{cfg.server.max_grad_norm}clip_constant_global_step_{cfg.server.constant_global_step}', str(similarity))
    elif cfg.tuning.parameter_to_tune == 'clipping':
        save_path = os.path.join(cfg.tuning.save_path+f'_global_step_{cfg.server.constant_global_step}', str(similarity), str(cfg.server.local_step))
    # Loading the results
    for client_ratio in cfg.results.client_ratios:
        client_ratios[client_ratio] = {}
        for hyperparameter in cfg.tuning.hyperparameter_grid:
            client_ratios[client_ratio][hyperparameter] = {}
            if cfg.tuning.parameter_to_tune == 'step_size':
                dir_path = os.path.join(save_path, f"{client_ratio}ur", f"{hyperparameter}beta")
            elif cfg.tuning.parameter_to_tune == 'clipping':
                dir_path = os.path.join(save_path, f"{client_ratio}ur", f"{hyperparameter}clipping")
            for fold in range(cfg.tuning.cv_folds):
                client_ratios[client_ratio][hyperparameter][fold] = {}
                file_name = f"fold_{fold}"
                print(f"Loading results for Client Ratio: {client_ratio}, Hyperparameter: {hyperparameter}, Fold: {fold}")
                with open(os.path.join(dir_path, f"{file_name}.csv"), mode='r') as file:
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
                    client_ratios[client_ratio][hyperparameter][fold]['test_accuracy'] = test_accuracies
                    client_ratios[client_ratio][hyperparameter][fold]['train_accuracy'] = train_accuracies
                    client_ratios[client_ratio][hyperparameter][fold]['train_loss'] = train_losses
                    client_ratios[client_ratio][hyperparameter][fold]['test_loss'] = test_losses

    if cfg.tuning.type=='cross_validation':
        perform_simple_cross_validation_analysis(cfg, client_ratios, similarity, evaluation_metrics, save_path)
    elif cfg.tuning.type=='early_stopping':
        perform_early_stopping_analysis(cfg, client_ratios, save_path)
                    


def tune_hyperparameters(cfg):
    set_seed(cfg.run_settings.seed)
    if cfg.dataset.name == 'synthetic':
        model = getattr(models, cfg.dataset.model_name)(input_dim=cfg.dataset.dim_input, output_dim=cfg.dataset.dim_output)
    else:
        model = getattr(models, cfg.dataset.model_name)()

    if cfg.dataset.name == 'synthetic':
        if cfg.dataset.iid:
            similarity = 'iid'
        else:
            similarity = (cfg.dataset.alpha, cfg.dataset.beta)
    else:
        similarity = cfg.dataset.similarity
    if cfg.tuning.type=='cross_validation':
        for hyperparameter in cfg.tuning.hyperparameter_grid:
            print(f"Tuning hyperparameter: {hyperparameter}")
            if cfg.tuning.parameter_to_tune == 'step_size':
                print(f"Tuning local step_size: {hyperparameter}")
                if cfg.server.constant_global_step == 'Fixed':
                    global_step = cfg.server.global_step
                    local_step = hyperparameter
                elif cfg.server.constant_global_step == 'Adaptive':
                    global_step = (cfg.server.client_ratio*cfg.dataset.nb_users)**0.5
                    local_step = hyperparameter/(cfg.server.local_updates*global_step)
                elif cfg.server.constant_global_step == 'Heuristic':
                    global_step = (cfg.server.client_ratio*cfg.dataset.nb_users)**0.5
                    local_step = hyperparameter*((cfg.server.client_ratio)**(2/3))/(cfg.server.local_updates*global_step)
                clipping_value = cfg.server.max_grad_norm
                save_path = os.path.join(cfg.tuning.save_path+f'_{clipping_value}clip_constant_global_step_${cfg.server.constant_global_step}', str(similarity), f"{cfg.server.client_ratio}ur", f"{hyperparameter}beta")
                os.makedirs(save_path, exist_ok=True)
            elif cfg.tuning.parameter_to_tune == 'clipping':
                print(f"Tuning clipping constant: {hyperparameter}")
                clipping_value = hyperparameter
                if cfg.server.constant_global_step == 'Fixed':
                    global_step = cfg.server.global_step
                    local_step = cfg.server.local_step
                elif cfg.server.constant_global_step == 'Adaptive':
                    global_step = (cfg.server.client_ratio*cfg.dataset.nb_users)**0.5
                    local_step = cfg.server.local_step/(cfg.server.local_updates*global_step)
                elif cfg.server.constant_global_step == 'Heuristic':
                    global_step = (cfg.server.client_ratio*cfg.dataset.nb_users)**0.5
                    local_step = cfg.server.local_step*((cfg.server.client_ratio)**(2/3))/(cfg.server.local_updates*global_step)
                save_path = os.path.join(cfg.tuning.save_path+f'_global_step_{cfg.server.constant_global_step}', str(similarity), str(cfg.server.local_step), f"{cfg.server.client_ratio}ur", f"{clipping_value}clipping")
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
                    max_grad_norm=clipping_value, 
                    x_label=cfg.dataset.x_label,
                    y_label=cfg.dataset.y_label, 
                    resume=cfg.run_settings.resume_from_checkpoint
                )
                server.train()
    else:
        pass

