import numpy as np
import os
import matplotlib.pyplot as plt
import csv

import math

def _sort_parameter_keys(keys, cfg):
    try:
        if cfg.results.transfer_mode == 'sigma':
            return sorted(keys, key=float)
        else:
            return sorted(keys, key=float, reverse=True)
    except Exception:
        return sorted(keys, key=str)

def plot_stacked_privacy_levels(
    cfg,
    loaded_results,
    evaluation_metrics,
    save_path,
    show_legend=True,
    panel_title=None,              # e.g. r"Tuning $\kappa$" or r"Tuning $C$"
    final_round_markers=None,      # e.g. [20, 40, 80]
    marker_labels=True,            # whether to annotate the vertical lines
):
    os.makedirs(save_path, exist_ok=True)

    # ------------------------------------------------------------
    # Plot settings
    # ------------------------------------------------------------
    FIG_WIDTH = 3.0
    ROW_HEIGHT = 1.55

    AXIS_LABEL_FONTSIZE = 8
    ROW_TITLE_FONTSIZE = 7
    PANEL_TITLE_FONTSIZE = 16
    TICK_FONTSIZE = 7
    LEGEND_FONTSIZE = 8
    LINE_WIDTH = 1.0

    TOP_MARGIN = 0.91

    parameter_values = _sort_parameter_keys(list(loaded_results.keys()), cfg)

    for fold in range(cfg.tuning.cv_folds):
        for metric in evaluation_metrics:
            n_rows = len(parameter_values)

            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=1,
                sharex=True,
                sharey=False,
                figsize=(FIG_WIDTH, max(ROW_HEIGHT * n_rows, 3.0))
            )

            if n_rows == 1:
                axes = [axes]

            for idx, (ax, parameter_varied) in enumerate(zip(axes, parameter_values)):
                for hyperparameter in cfg.tuning.hyperparameter_grid:
                    if hyperparameter not in loaded_results[parameter_varied]:
                        continue

                    y = loaded_results[parameter_varied][hyperparameter][fold][metric]
                    ax.plot(y, label=f"{hyperparameter}", linewidth=LINE_WIDTH)

                # Add vertical lines only to the first subplot (final training setting)
                if idx == 0 and final_round_markers is not None:
                    for t_marker in final_round_markers:
                        ax.axvline(
                            x=t_marker,
                            color="black",
                            linestyle="--",
                            linewidth=0.8,
                            alpha=0.7
                        )

                        if marker_labels:
                            ymin, ymax = ax.get_ylim()
                            ax.text(
                                t_marker,
                                ymax,
                                f"{t_marker}",
                                ha="center",
                                va="bottom",
                                fontsize=6,
                                rotation=90,
                                color="black"
                            )

                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=AXIS_LABEL_FONTSIZE)
                ax.text(
                    0.98, 0.95,
                    f"{cfg.results.transfer_mode} = {parameter_varied}",
                    transform=ax.transAxes,
                    ha="right", va="top",
                    fontsize=ROW_TITLE_FONTSIZE
                )
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

            axes[-1].set_xlabel("Communication Round", fontsize=AXIS_LABEL_FONTSIZE)

            # ------------------------------------------------------------
            # Figure-level title first, legend below it
            # If show_legend=False, neither title nor legend is shown
            # ------------------------------------------------------------
            if show_legend:
                if panel_title is not None:
                    fig.text(
                        0.5, 0.985,
                        panel_title,
                        ha="center", va="top",
                        fontsize=PANEL_TITLE_FONTSIZE
                    )

                handles, labels = axes[0].get_legend_handles_labels()
                if handles:
                    fig.legend(
                        handles,
                        labels,
                        title=None,
                        loc="upper center",
                        ncol=len(labels),
                        fontsize=LEGEND_FONTSIZE,
                        bbox_to_anchor=(0.5, 0.965),
                        frameon=False,
                        columnspacing=0.35,
                        handlelength=1.0,
                        handletextpad=0.2,
                        borderaxespad=0.0
                    )

            fig.subplots_adjust(
                top=TOP_MARGIN,
                hspace=0.14,
                left=0.22,
                right=0.97,
                bottom=0.06
            )

            legend_tag = "with_legend" if show_legend else "no_legend"
            marker_tag = "with_Tmarkers" if final_round_markers is not None else "no_Tmarkers"

            fig.savefig(
                os.path.join(
                    save_path,
                    f"{cfg.results.transfer_mode}_stacked_{metric.lower().replace(' ', '_')}_{legend_tag}_{marker_tag}_fold_{fold}.png"
                ),
                dpi=300
            )
            plt.close(fig)


def perform_early_stopping_analysis(cfg, loaded_results, save_path):
    os.makedirs(save_path, exist_ok=True)
    if cfg.tuning.early_stopping_resource == "rounds":
        print("Performing early stopping analysis based on rounds...")
        for parameter_varied in loaded_results:
            with open(os.path.join(save_path, f"early_stopping_summary_{parameter_varied}_{cfg.results.transfer_mode}.txt"), mode='w', newline='') as file:
                file.write(f"Starting early stopping analysis for client ratio: {parameter_varied}\n")
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
                with open(os.path.join(save_path, f"early_stopping_summary_{parameter_varied}_{cfg.results.transfer_mode}.txt"), mode='a', newline='') as file:
                    file.write(f"Stage {stage}: Evaluating up to round {end} with candidate hyperparameters: {candidate_hyperparameters}\n")
                    for hyperparameter in candidate_hyperparameters:
                        if cfg.tuning.metric == 'train_loss':
                            intermediate_results[hyperparameter] = np.mean([min(loaded_results[parameter_varied][hyperparameter][fold]['train_loss'][:end]) for fold in range(cfg.tuning.cv_folds)])
                            file.write(f"Hyperparameter: {hyperparameter}, Intermediate Train Loss: {intermediate_results[hyperparameter]}\n")
                            results[hyperparameter] = np.mean([np.array(loaded_results[parameter_varied][hyperparameter][fold]['train_loss'][:end]) for fold in range(cfg.tuning.cv_folds)], axis=0)
                        elif cfg.tuning.metric == 'train_accuracy':
                            intermediate_results[hyperparameter] = np.mean([max(loaded_results[parameter_varied][hyperparameter][fold]['train_accuracy'][:end]) for fold in range(cfg.tuning.cv_folds)])
                            file.write(f"Hyperparameter: {hyperparameter}, Intermediate Train Accuracy: {intermediate_results[hyperparameter]}\n")
                            results[hyperparameter] = np.mean([np.array(loaded_results[parameter_varied][hyperparameter][fold]['train_accuracy'][:end]) for fold in range(cfg.tuning.cv_folds)], axis=0)
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
            plt.title(f"{cfg.tuning.metric.replace('_', ' ').title()} vs Communication Rounds for {cfg.results.transfer_mode}: {parameter_varied}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"early_stopping_{cfg.tuning.metric.lower().replace(' ', '_')}_analysis_{cfg.results.transfer_mode}_{parameter_varied}.png"))
            plt.close()


    elif cfg.tuning.early_stopping_resource == "client_sampling":
        print("Performing early stopping analysis based on client sampling...")
        raise NotImplementedError("Early stopping based on client sampling is not implemented yet.")

def perform_simple_cross_validation_analysis(cfg, loaded_results, similarity, evaluation_metrics, save_path):
    print("Performing simple cross-validation analysis...")
    os.makedirs(save_path, exist_ok=True)
    for parameter_varied in loaded_results:
        best_hyperparameter = None
        best_accuracy = 0.0
        best_loss = float('inf')
        for hyperparameter in loaded_results[parameter_varied]:
            best_test_accuracy = []
            best_train_accuracy = []
            best_train_loss = []
            best_test_loss = []
            for fold in range(cfg.tuning.cv_folds):
                best_test_accuracy.append(max(loaded_results[parameter_varied][hyperparameter][fold]['test_accuracy']))
                best_train_accuracy.append(max(loaded_results[parameter_varied][hyperparameter][fold]['train_accuracy']))
                best_train_loss.append(min(loaded_results[parameter_varied][hyperparameter][fold]['train_loss']))
                best_test_loss.append(min(loaded_results[parameter_varied][hyperparameter][fold]['test_loss']))
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
            loaded_results[parameter_varied][hyperparameter][fold]['avg_best_test_accuracy'] = round(avg_best_test_accuracy, 4)
            loaded_results[parameter_varied][hyperparameter][fold]['avg_best_train_accuracy'] = round(avg_best_train_accuracy, 4)
            loaded_results[parameter_varied][hyperparameter][fold]['std_train_accuracy'] = round(std_best_train_accuracy, 4)
            loaded_results[parameter_varied][hyperparameter][fold]['std_test_accuracy'] = round(std_best_test_accuracy, 4)
            loaded_results[parameter_varied][hyperparameter][fold]['avg_best_train_loss'] = round(avg_best_train_loss, 4)
            loaded_results[parameter_varied][hyperparameter][fold]['avg_best_test_loss'] = round(avg_best_test_loss, 4)
        print('Writing tuning summary to file and plotting results...')
            # Plot and write it to a file
        for fold in range(cfg.tuning.cv_folds):
            with open(os.path.join(save_path, f"tuning_summary_{fold}_{parameter_varied}_{cfg.results.transfer_mode}.txt"), mode='w', newline='') as file:
                for hyperparameter in loaded_results[parameter_varied]:
                    metrics = loaded_results[parameter_varied][hyperparameter][fold]
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
                        plt.plot(loaded_results[parameter_varied][hyperparameter][fold]['train_accuracy'], label=f"Beta: {hyperparameter}")
                    elif metric == 'test_accuracy':
                        plt.plot(loaded_results[parameter_varied][hyperparameter][fold]['test_accuracy'], label=f"Beta: {hyperparameter}")
                    elif metric == 'train_loss':
                        plt.plot(loaded_results[parameter_varied][hyperparameter][fold]['train_loss'], label=f"Beta: {hyperparameter}")
                    elif metric == 'test_loss':
                        plt.plot(loaded_results[parameter_varied][hyperparameter][fold]['test_loss'], label=f"Beta: {hyperparameter}")
                ax = plt.gca()
                round_markers = [5,17,38,65,95,150]
                for t_marker in round_markers:
                    ax.axvline(
                        x=t_marker,
                        color="black",
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.7
                    )
                ax.set_xticks(round_markers)
                ax.set_xlabel("Communication Round")
                ax.set_ylabel(metric)
                ax.set_title(f"{metric} vs Communication Rounds")
                ax.legend(title=None)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        save_path,
                        f"{metric.lower().replace(' ', '_')}_comparison_{fold}_{parameter_varied}_{cfg.results.transfer_mode}.png"
                    ),
                    dpi=300
                )
                plt.close()

    for fold in range(cfg.tuning.cv_folds):
        for metric in evaluation_metrics:
            for hyperparameter in cfg.tuning.hyperparameter_grid:
                plt.figure()
                for parameter_varied in loaded_results:
                    plt.plot(loaded_results[parameter_varied][hyperparameter][fold][metric], label=f"{cfg.results.transfer_mode}: {parameter_varied}")
                plt.xlabel("Communication Round")
                plt.ylabel(metric)
                plt.title(f"{metric} vs Communication Rounds for Hyperparameter: {hyperparameter}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f"{cfg.results.transfer_mode}_comparison_{metric.lower().replace(' ', '_')}_hyperparameter_{hyperparameter}_{fold}.png"))
                plt.close()
    
    plot_stacked_privacy_levels(
        cfg,
        loaded_results,
        evaluation_metrics=["train_loss"],
        save_path=save_path,
        show_legend=True if cfg.results.transfer_mode == 'sigma' else False,
        panel_title=r"Tuning $\kappa$" if cfg.tuning.parameter_to_tune == 'step_size' else r"Tuning $C$",
        final_round_markers=[],
        marker_labels=True
    )

def load_results(cfg, save_path, similarity):
    loaded_results = {}
    # Loading the results
    for parameter_varied in cfg.results.transfer_parameters:
        loaded_results[parameter_varied] = {}
        for hyperparameter in cfg.tuning.hyperparameter_grid:
            loaded_results[parameter_varied][hyperparameter] = {}
            if cfg.results.transfer_mode == 'client_ratio':
                if cfg.tuning.parameter_to_tune == 'step_size':
                    dir_path = os.path.join(save_path, f"{parameter_varied}ur", f"{hyperparameter}beta")
                elif cfg.tuning.parameter_to_tune == 'clipping':
                    dir_path = os.path.join(save_path, f"{parameter_varied}ur", f"{hyperparameter}clipping")
            elif cfg.results.transfer_mode == 'sigma':
                if cfg.tuning.parameter_to_tune == 'step_size':
                    dir_path = os.path.join(save_path+f"_{parameter_varied}sigma_{cfg.server.max_grad_norm}clip_constant_global_step_{cfg.server.constant_global_step}", str(similarity), f"{cfg.server.client_ratio}ur",f"{hyperparameter}beta")
                elif cfg.tuning.parameter_to_tune == 'clipping':
                    dir_path = os.path.join(save_path+f'_{parameter_varied}sigma_global_step_{cfg.server.constant_global_step}', str(similarity), str(cfg.server.local_step), f"{cfg.server.client_ratio}ur", f"{hyperparameter}clipping")
            elif cfg.results.transfer_mode == 'rounds':
                if cfg.tuning.parameter_to_tune == 'step_size':
                    dir_path = os.path.join(save_path, f"{cfg.server.client_ratio}ur", f"{hyperparameter}beta")
                elif cfg.tuning.parameter_to_tune == 'clipping':
                    dir_path = os.path.join(save_path, f"{cfg.server.client_ratio}ur", f"{hyperparameter}clipping")

            elif cfg.results.transfer_mode == 'sampling_rate':
                if cfg.tuning.parameter_to_tune == 'step_size':
                    dir_path = os.path.join(save_path+f"_{parameter_varied}sr_{cfg.server.dp}dp_{cfg.server.sigma}sigma_{cfg.server.max_grad_norm}clip_constant_global_step_{cfg.server.constant_global_step}", str(similarity), f"{cfg.server.client_ratio}ur", f"{hyperparameter}beta")
                elif cfg.tuning.parameter_to_tune == 'clipping':
                    dir_path = os.path.join(save_path+f"_{parameter_varied}sr_{cfg.server.dp}dp_{cfg.server.sigma}sigma_global_step_{cfg.server.constant_global_step}", str(similarity), str(cfg.server.local_step), f"{cfg.server.client_ratio}ur", f"{hyperparameter}clipping")
            
            elif cfg.results.transfer_mode == 'local_updates':
                if cfg.tuning.parameter_to_tune == 'step_size':
                    dir_path = os.path.join(save_path+f"_{parameter_varied}K_{cfg.server.sampling_rate}sr_{cfg.server.dp}dp_{cfg.server.sigma}sigma_{cfg.server.max_grad_norm}clip_constant_global_step_{cfg.server.constant_global_step}", str(similarity), f"{cfg.server.client_ratio}ur", f"{hyperparameter}beta")
                elif cfg.tuning.parameter_to_tune == 'clipping':
                    dir_path = os.path.join(save_path+f"_{parameter_varied}K_{cfg.server.sampling_rate}sr_{cfg.server.dp}dp_{cfg.server.sigma}sigma_global_step_{cfg.server.constant_global_step}", str(similarity), str(cfg.server.local_step), f"{cfg.server.client_ratio}ur", f"{hyperparameter}clipping")
            print(f"Loading results from directory: {dir_path}")
            for fold in range(cfg.tuning.cv_folds):
                loaded_results[parameter_varied][hyperparameter][fold] = {}
                file_name = f"fold_{fold}"
                print(f"Loading results for {cfg.results.transfer_mode}: {parameter_varied}, Hyperparameter: {hyperparameter}, Fold: {fold}")
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
                    if cfg.results.transfer_mode == 'rounds':
                        test_accuracies = test_accuracies[:int(parameter_varied)]
                        train_accuracies = train_accuracies[:int(parameter_varied)]
                        train_losses = train_losses[:int(parameter_varied)]
                        test_losses = test_losses[:int(parameter_varied)]
                    loaded_results[parameter_varied][hyperparameter][fold]['test_accuracy'] = test_accuracies
                    loaded_results[parameter_varied][hyperparameter][fold]['train_accuracy'] = train_accuracies
                    loaded_results[parameter_varied][hyperparameter][fold]['train_loss'] = train_losses
                    loaded_results[parameter_varied][hyperparameter][fold]['test_loss'] = test_losses
    return loaded_results
