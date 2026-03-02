import numpy as np
import os
import matplotlib.pyplot as plt

def perform_early_stopping_analysis(cfg, client_ratios, save_path):
    if cfg.tuning.early_stopping_resource == "rounds":
        print("Performing early stopping analysis based on rounds...")
        for client_ratio in client_ratios:
            with open(os.path.join(save_path, f"{client_ratio}ur", f"early_stopping_stage.txt"), mode='w', newline='') as file:
                file.write(f"Starting early stopping analysis for client ratio: {client_ratio}\n")
            results = {}
            stage = 1
            candidate_hyperparameters = cfg.tuning.hyperparameter_grid.copy()
            end = 0
            while end < cfg.run_settings.rounds and len(candidate_hyperparameters) > 1:
                intermediate_results = {}
                if end+(2**(stage-1)) * cfg.tuning.min_resource < cfg.run_settings.rounds and round(len(candidate_hyperparameters)/cfg.tuning.elimination_rate) > 2:
                    end += (2**(stage-1)) * cfg.tuning.min_resource
                else:
                    end = cfg.run_settings.rounds
                with open(os.path.join(save_path, f"{client_ratio}ur", f"early_stopping_stage.txt"), mode='a', newline='') as file:
                    file.write(f"Stage {stage}: Evaluating up to round {end} with candidate hyperparameters: {candidate_hyperparameters}\n")
                    for hyperparameter in candidate_hyperparameters:
                        if cfg.tuning.metric == 'train_loss':
                            intermediate_results[hyperparameter] = np.mean([min(client_ratios[client_ratio][hyperparameter][fold]['train_loss'][:end]) for fold in range(cfg.tuning.cv_folds)])
                            file.write(f"Hyperparameter: {hyperparameter}, Intermediate Train Loss: {intermediate_results[hyperparameter]}\n")
                            results[hyperparameter] = np.mean([np.array(client_ratios[client_ratio][hyperparameter][fold]['train_loss'][:end]) for fold in range(cfg.tuning.cv_folds)], axis=0)
                        elif cfg.tuning.metric == 'train_accuracy':
                            intermediate_results[hyperparameter] = np.mean([max(client_ratios[client_ratio][hyperparameter][fold]['train_accuracy'][:end]) for fold in range(cfg.tuning.cv_folds)])
                            file.write(f"Hyperparameter: {hyperparameter}, Intermediate Train Accuracy: {intermediate_results[hyperparameter]}\n")
                            results[hyperparameter] = np.mean([np.array(client_ratios[client_ratio][hyperparameter][fold]['train_accuracy'][:end]) for fold in range(cfg.tuning.cv_folds)], axis=0)
                if cfg.tuning.metric == 'train_loss':
                    candidate_hyperparameters = dict(sorted(intermediate_results.items(), key=lambda item: item[1])[:round(len(candidate_hyperparameters)/cfg.tuning.elimination_rate)]).keys()
                elif cfg.tuning.metric == 'train_accuracy':
                    candidate_hyperparameters = dict(sorted(intermediate_results.items(), key=lambda item: item[1], reverse=True)[:round(len(candidate_hyperparameters)/cfg.tuning.elimination_rate)]).keys()
                stage += 1
            plt.figure()
            for hyperparameter in cfg.tuning.hyperparameter_grid:
                plt.plot(results[hyperparameter], label=f"{cfg.tuning.parameter_to_tune}: {hyperparameter}")
            plt.xlabel("Communication Round")
            plt.ylabel(cfg.tuning.metric.replace('_', ' ').title())
            plt.title(f"{cfg.tuning.metric.replace('_', ' ').title()} vs Communication Rounds for Client Ratio: {client_ratio}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"{client_ratio}ur", f"early_stopping_{cfg.tuning.metric.lower().replace(' ', '_')}_analysis.png"))
            plt.close()


    elif cfg.tuning.early_stopping_resource == "client_sampling":
        print("Performing early stopping analysis based on client sampling...")
        raise NotImplementedError("Early stopping based on client sampling is not implemented yet.")

def perform_simple_cross_validation_analysis(cfg, client_ratios, similarity, evaluation_metrics, save_path):
    print("Performing simple cross-validation analysis...")
    for client_ratio in client_ratios:
        best_hyperparameter = None
        best_accuracy = 0.0
        best_loss = float('inf')
        for hyperparameter in client_ratios[client_ratio]:
            best_test_accuracy = []
            best_train_accuracy = []
            best_train_loss = []
            best_test_loss = []
            for fold in range(cfg.tuning.cv_folds):
                best_test_accuracy.append(max(client_ratios[client_ratio][hyperparameter][fold]['test_accuracy']))
                best_train_accuracy.append(max(client_ratios[client_ratio][hyperparameter][fold]['train_accuracy']))
                best_train_loss.append(min(client_ratios[client_ratio][hyperparameter][fold]['train_loss']))
                best_test_loss.append(min(client_ratios[client_ratio][hyperparameter][fold]['test_loss']))
            avg_best_test_accuracy = np.mean(best_test_accuracy)
            avg_best_train_accuracy = np.mean(best_train_accuracy)
            avg_best_train_loss = np.mean(best_train_loss)
            avg_best_test_loss = np.mean(best_test_loss)
            std_best_test_accuracy = np.std(best_test_accuracy)
            std_best_train_accuracy = np.std(best_train_accuracy)
            std_best_train_loss = np.std(best_train_loss)
            std_best_test_loss = np.std(best_test_loss)
            if cfg.tuning.metric == 'train_loss' and avg_best_train_loss < best_loss:
                best_loss = avg_best_train_loss
                best_hyperparameter = hyperparameter
            elif cfg.tuning.metric == 'train_accuracy' and avg_best_train_accuracy > best_accuracy:
                best_accuracy = avg_best_train_accuracy
                best_hyperparameter = hyperparameter
            client_ratios[client_ratio][hyperparameter][fold]['avg_best_test_accuracy'] = round(avg_best_test_accuracy, 4)
            client_ratios[client_ratio][hyperparameter][fold]['avg_best_train_accuracy'] = round(avg_best_train_accuracy, 4)
            client_ratios[client_ratio][hyperparameter][fold]['std_train_accuracy'] = round(std_best_train_accuracy, 4)
            client_ratios[client_ratio][hyperparameter][fold]['std_test_accuracy'] = round(std_best_test_accuracy, 4)
            client_ratios[client_ratio][hyperparameter][fold]['avg_best_train_loss'] = round(avg_best_train_loss, 4)
            client_ratios[client_ratio][hyperparameter][fold]['avg_best_test_loss'] = round(avg_best_test_loss, 4)
        print('Writing tuning summary to file and plotting results...')
            # Plot and write it to a file
        for fold in range(cfg.tuning.cv_folds):
            with open(os.path.join(save_path, f"{client_ratio}ur", f"tuning_summary_{fold}.txt"), mode='w', newline='') as file:
                for hyperparameter in client_ratios[client_ratio]:
                    metrics = client_ratios[client_ratio][hyperparameter][fold]
                    file.write(f"Hyperparameter: {hyperparameter}\n")
                    file.write(f"Avg Best Train Accuracy: {metrics['avg_best_train_accuracy']}\n")
                    file.write(f"Std Train Accuracy: {metrics['std_train_accuracy']}\n")
                    file.write(f"Avg Best Test Accuracy: {metrics['avg_best_test_accuracy']}\n")
                    file.write(f"Std Test Accuracy: {metrics['std_test_accuracy']}\n")
                    file.write(f"Avg Best Train Loss: {metrics['avg_best_train_loss']}\n")
                    file.write(f"Avg Best Test Loss: {metrics['avg_best_test_loss']}\n")
                    file.write("\n")
                if cfg.tuning.metric == 'train_loss':
                    file.write(f"Best Hyperparameter: {best_hyperparameter} with Avg Best {cfg.tuning.metric}: {round(best_loss, 4)}\n")
                elif cfg.tuning.metric == 'train_accuracy':
                    file.write(f"Best Hyperparameter: {best_hyperparameter} with Avg Best {cfg.tuning.metric}: {round(best_accuracy, 4)}\n")
            for metric in evaluation_metrics:
                plt.figure()
                for hyperparameter in cfg.tuning.hyperparameter_grid:
                    if metric == 'train_accuracy':
                        plt.plot(client_ratios[client_ratio][hyperparameter][fold]['train_accuracy'], label=f"Beta: {hyperparameter}")
                    elif metric == 'test_accuracy':
                        plt.plot(client_ratios[client_ratio][hyperparameter][fold]['test_accuracy'], label=f"Beta: {hyperparameter}")
                    elif metric == 'train_loss':
                        plt.plot(client_ratios[client_ratio][hyperparameter][fold]['train_loss'], label=f"Beta: {hyperparameter}")
                    elif metric == 'test_loss':
                        plt.plot(client_ratios[client_ratio][hyperparameter][fold]['test_loss'], label=f"Beta: {hyperparameter}")
                plt.xlabel("Communication Round")
                plt.ylabel(metric)
                plt.title(f"{metric} vs Communication Rounds")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"{client_ratio}ur", f"{metric.lower().replace(' ', '_')}_comparison_{fold}.png"))
                plt.close()

    for fold in range(cfg.tuning.cv_folds):
        for metric in evaluation_metrics:
            for hyperparameter in cfg.tuning.hyperparameter_grid:
                plt.figure()
                for client_ratio in cfg.results.client_ratios:
                    plt.plot(client_ratios[client_ratio][hyperparameter][fold][metric], label=f"Client Ratio: {client_ratio}")
                plt.xlabel("Communication Round")
                plt.ylabel(metric)
                plt.title(f"{metric} vs Communication Rounds for Hyperparameter: {hyperparameter}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"client_ratio_comparison_{metric.lower().replace(' ', '_')}_hyperparameter_{hyperparameter}_{fold}.png"))
                plt.close()
                    
