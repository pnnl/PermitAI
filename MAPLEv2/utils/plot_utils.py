import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from scipy import stats
from pathlib import Path

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

from utils.logging_utils import LogManager

if __name__ == "__main__":
    LogManager.initialize("logs/plot_utils.log")

logger = LogManager.get_logger(__name__)

class HeatmapGenerator:
    @staticmethod
    def compute_closed_question_correctness(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the correct answer correctness values for closed questions.
        Comparison is case-insensitive.
        
        Args:
            df (pd.DataFrame): DataFrame containing 'question_type', 'answer_expected', and 'answer_predicted' columns.
        
        Returns:
            pd.DataFrame: DataFrame with corrected 'answer_correctness' values for closed questions.
        """
        if 'question_type' in df.columns and 'answer_expected' in df.columns and 'answer_predicted' in df.columns:
            mask = df['question_type'] == 'closed'
            df.loc[mask, 'answer_correctness'] = np.where(
                (df.loc[mask, 'answer_expected'].str.lower() == df.loc[mask, 'answer_predicted'].str.lower()),
                1.0,
                0.0
            )
            return df

        else:
            logger.warning("Some necessary columns are missing in the output dataframe.")
            return df

    @classmethod
    def generate_matrix_heatmap_multiple_context(
        cls, path_pfx: str, model: str, model_name: str, 
        context: List[str], metric: str, output_path: str, 
        **kwargs):
        """
        Generate a matrix heatmap for a given metric across different context types and question types.

        Args:
            path_pfx (str): Common prefix for CSV file paths.
            model (str): Model name used in file path
            model_name (str): Model name to be used in title
            context (List[str]): List of context types to process.
            metric (str): The metric to plot.
            output_path (str): Path to save the output heatmap.
        """
        logger.info(f"Generating matrix heatmap for metric: {metric}")

        # Keyword arguments
        figsize = kwargs.get("figsize", (12,6))
        labelsize = kwargs.get("labelsize", 18)
        ticklabelsize = kwargs.get("ticklabelsize", 15)
        xtickrotation = kwargs.get("xtickrotation", 0)
        annotation_size = kwargs.get("annotation_size", 15)
        colormap = kwargs.get("colormap", "Blues")
        in_percent = kwargs.get("percentage", False)

        # Initialize an empty list to store DataFrames
        dfs = []

        # Process each context type
        for context_type in context:
            try:
                file_path = f"{path_pfx}_{model}/scores_{context_type}.json"
                with open(file_path, 'r') as f:
                    scores = json.load(f)
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                continue

            try:
                # Extract relevant data
                scores_dict = {'question_type': [], metric: []}
                for key, entry in scores.items():
                    if metric not in entry['scores']:
                        logger.error(f"Metric '{metric}' not found for entry in file: {file_path}")
                        continue

                    scores_dict['question_type'].append(entry['question_type'])
                    score = 0.0 if not entry['scores'][metric] else entry['scores'][metric]
                    if entry['question_type'] == 'closed' and metric == 'answer_correctness':
                        # Compute correctness for closed questions
                        correct = 1.0 if entry['answer_expected'].lower() == entry['answer_predicted'].lower() else 0.0
                        scores_dict[metric].append(correct * 100.0 if in_percent else correct)
                    else:
                        scores_dict[metric].append(score * 100.0 if in_percent else score)
                
                df = pd.DataFrame(scores_dict)

            except Exception as e:
                logger.error(f"Error preparing dataframe: {str(e)}")
                continue
            
            try:
                # Group by question_type and compute mean
                grouped = df.groupby('question_type')[metric].mean().reset_index()
                grouped['context_type'] = context_type
                dfs.append(grouped)
            except Exception as e:
                logger.error(f"Error processing dataframe: {str(e)}")

        if not dfs:
            logger.error("No valid data to plot. Exiting.")
            return

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Pivot the DataFrame to create the matrix
        matrix_df = combined_df.pivot(index='question_type', columns='context_type', values=metric)

        # Reorder columns based on the input context list
        matrix_df = matrix_df.reindex(columns=context)

        # Create the heatmap
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if in_percent:
            vmin = 0.0
            vmax = 100.0
        else:
            vmin = 0.0
            vmax = 1.0
        ax = sns.heatmap(
            matrix_df, annot=True, cmap=colormap, fmt='.2f', 
            annot_kws={"size": annotation_size},
            vmin=vmin, vmax=vmax, ax=ax
            )
        
        cbar = ax.collections[0].colorbar
        if in_percent:
            cbar.ax.set_ylabel(
                f"{metric.replace('_',' ').title()}(%)", 
                fontsize=labelsize
                )
        else:
            cbar.ax.set_ylabel(
                metric.replace('_',' ').title(), 
                fontsize=labelsize
                )
        cbar.ax.tick_params(
            axis='y', 
            labelsize=ticklabelsize
            )

        
        ax.set_ylabel(
            "Question Type", 
            fontsize = labelsize
            )
        ax.set_xlabel(
            "Context Type", 
            fontsize = labelsize
            )
        ax.set_title(
            f"Model: {model_name}", 
            fontsize=labelsize
            )

        # Rotate x-axis labels for better readability
        ax.tick_params(
            axis='x', 
            rotation=xtickrotation, 
            labelsize=ticklabelsize
            )
        ax.tick_params(axis='y', labelsize=ticklabelsize)

        # Adjust layout and save
        dir = Path(output_path).parent
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
            os.chmod(dir, 0o777)
        if not output_path.lower().endswith('.png'):
            output_path += '.png'
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Multi-context heatmap saved to {output_path}")

    @classmethod
    def generate_matrix_heatmap_single_context(cls, path_pfx: str, models: List[str], metric: str, output_path: str, context_type: str = "gold", **kwargs):
        """
        Generate a matrix heatmap for a given metric across different context types and question types.

        Args:
            path_pfx (str): Common prefix for CSV file paths.
            models (List[str]): List of models to process.
            metric (str): The metric to plot.
            output_path (str): Path to save the output heatmap.
            context_type (str): The type of context to process
        """

        # Keyword arguments
        figsize = kwargs.get("figsize", (12,6))
        labelsize = kwargs.get("labelsize", 18)
        ticklabelsize = kwargs.get("ticklabelsize", 15)
        xticklabels = kwargs.get("xticklabels", None)
        xtickrotation = kwargs.get("xtickrotation", 0)
        annotation_size = kwargs.get("annotation_size", 15)
        colormap = kwargs.get("colormap", "Blues")
        in_percent = kwargs.get("percentage", False)

        logger.info(f"Generating matrix heatmap for metric: {metric}")

        # Initialize an empty list to store DataFrames
        dfs = []

        # Process each context type
        for model in models:
            file_path = f"{path_pfx}_{model}/scores_{context_type}.csv"
            try:
                df = pd.read_csv(file_path)
                if metric not in df.columns:
                    logger.error(f"Metric '{metric}' not found in file: {file_path}")
                    continue

                # If the metric is answer_correctness, compute correct values for closed questions
                if metric == 'answer_correctness':
                    df = cls.compute_closed_question_correctness(df)
                
                # Check if required to show plot in percentage
                if in_percent:
                    df[metric] = df[metric] * 100.0

                # Group by question_type and compute mean
                grouped = df.groupby('question_type')[metric].mean().reset_index()
                grouped['model'] = model
                dfs.append(grouped)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")

        if not dfs:
            logger.error("No valid data to plot. Exiting.")
            return

        # Combine all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)

        # Pivot the DataFrame to create the matrix
        matrix_df = combined_df.pivot(index='question_type', columns='model', values=metric)

        # Reorder columns based on the input context list
        matrix_df = matrix_df.reindex(columns=models)

        # Create the heatmap
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if in_percent:
            vmin = 0.0
            vmax = 100.0
        else:
            vmin = 0.0
            vmax = 1.0
        ax = sns.heatmap(
            matrix_df, annot=True, cmap=colormap, fmt='.2f', 
            annot_kws={"size": annotation_size},
            vmin=vmin, vmax=vmax, ax=ax
            )
        cbar = ax.collections[0].colorbar
        if in_percent:
            cbar.ax.set_ylabel(
                f"{metric.replace('_',' ').title()}(%)", 
                fontsize=labelsize
                )
        else:
            cbar.ax.set_ylabel(
                metric.replace('_',' ').title(), 
                fontsize=labelsize
                )
        cbar.ax.tick_params(
            axis='y', 
            labelsize=ticklabelsize
            )

        
        ax.set_ylabel(
            kwargs.get("ylabel", "Question Type"), 
            fontsize = labelsize
            )
        ax.set_xlabel(
            kwargs.get("xlabel", "Pre-trained / Finetuned Models"), 
            fontsize = labelsize
            )

        # Rotate x-axis labels for better readability
        ax.tick_params(
            axis='y', 
            labelsize=ticklabelsize
            )
        
        if xticklabels:
            ax.set_xticklabels(
                xticklabels, rotation=xtickrotation, 
                fontsize=ticklabelsize
                )
        else:
            ax.tick_params(
                axis='x', rotation=xtickrotation, 
                labelsize=ticklabelsize
                )
        
        # Super title for the figure
        fig.suptitle(
            f"{metric.replace('_',' ').title()} heatmap for different LLMs", 
            fontsize=labelsize+2
            )

        # Adjust layout and save
        dir = Path(output_path).parent
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
            os.chmod(dir, 0o777)
        if not output_path.lower().endswith('.png'):
            output_path += '.png'
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Single context heatmap saved to {output_path}")


class BarPlotGenerator:
    
    @classmethod
    def generate_bar_plot(cls, path_pfx, models, metric="answer_correctness", context_type="gold", output_path="barplot_score_mean.png", **kwargs):
        
        # Keyword arguments
        figsize = kwargs.get("figsize", (12,6))
        labelsize = kwargs.get("labelsize", 18)
        ticklabelsize = kwargs.get("ticklabelsize", 15)
        xticklabels = kwargs.get("xticklabels", None)
        xtickrotation = kwargs.get("xtickrotation", 0)
        annotation_size = kwargs.get("annotation_size", 15)
        in_percent = kwargs.get("percentage", False)
        
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
        data_dict = {"model":[], "score":[]}
        for model in models:
            file_path = f"{path_pfx}_{model}/scores_{context_type}.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if metric in df.columns:
                    for _,row in df.iterrows():
                        data_dict["model"].append(model)
                        # Check if required to show plot in percentage
                        if in_percent:
                            data_dict["score"].append(row[metric] * 100.0)
                        else:
                            data_dict["score"].append(row[metric])

        if not data_dict:
            raise ValueError("No valid data found for the specified metric and models.")
        plot_df = pd.DataFrame(data_dict)

        # Run T-statistic test
        print_model_pairwise_ttest(plot_df, models, xticklabels)
        
        # Calculate t-statistics for each model
        overall_mean = plot_df['score'].mean()
        t_stats = {}
        p_val = {}
        for model in plot_df['model'].unique():
            model_scores = plot_df[plot_df['model'] == model]['score']
            model_scores = model_scores.dropna()
            t_stat, p_value = stats.ttest_1samp(model_scores, overall_mean)
            t_stats[model] = t_stat
            p_val[model] = p_value
        
        # Calculate t-statistics for each model
        overall_mean = plot_df['score'].mean()
        t_stats = {}
        p_val = {}
        for model in plot_df['model'].unique():
            model_scores = plot_df[plot_df['model'] == model]['score']
            model_scores = model_scores.dropna()
            t_stat, p_value = stats.ttest_1samp(model_scores, overall_mean)
            t_stats[model] = t_stat
            p_val[model] = p_value
        
        colors = plt.cm.rainbow(np.linspace(0, 1, 10))
        ax = sns.barplot(
            data=plot_df, 
            x="model", y="score", 
            hue="model", capsize=0.2, 
            palette=colors[:len(models)].tolist(), 
            ax=ax
            )
        
        if in_percent:
            ax.set_ylabel(
                kwargs.get("ylabel", f"{metric.replace('_',' ').title()} Scores (%)"), 
                fontsize = labelsize
                )
        else:
            ax.set_ylabel(
                kwargs.get("ylabel", f"{metric.replace('_',' ').title()} Scores"), 
                fontsize = labelsize
                )
        ax.set_xlabel(
            kwargs.get("xlabel", "Pre-trained / Finetuned Models"), 
            fontsize = labelsize
            )

        # Rotate x-axis labels for better readability
        ax.tick_params(
            axis='y', 
            labelsize=ticklabelsize
            )
        
        if xticklabels:
            ax.set_xticks(
                range(len(xticklabels)),
                xticklabels, rotation=xtickrotation, 
                fontsize=ticklabelsize
                )
        else:
            ax.tick_params(
                axis='x', rotation=kwargs.get("xtickrotation", 0), 
                labelsize=ticklabelsize
                )
        
        # Calculate means for each model
        means = [plot_df[plot_df['model'] == model]['score'].mean() for model in models]
        
        # Create labels with both mean and t-statistic
        labels = [f"$\\mu$={mean:.2f}\nt={t_stats[model]:.2f}\np={p_val[model]:.3f}" 
                for mean, model in zip(means, models)]
        
        # Add labels with both mean and t-statistic
        for idx, bar in enumerate(ax.patches):
            # Get the height of the bar
            height = bar.get_height()
            # Position the text in the middle of the bar
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height/2.,  # middle of the bar
                labels[idx],
                ha='center',
                va='center',
                fontsize=annotation_size
            )
        
        # Adjust layout and save
        dir = Path(output_path).parent
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
            os.chmod(dir, 0o777)
            
        if not output_path.lower().endswith('.png'):
            output_path += '.png'
        
        fig.suptitle(
            f"{metric.replace('_',' ').title()} Scores for different finetuned models", 
            fontsize=labelsize+2
            )
        fig.savefig(output_path, bbox_inches="tight")
        
        logger.info(f"Bar plot saved to {output_path}")


def print_model_pairwise_ttest(df, models, modelnames, paired=True):
    """
    Perform unpaired t-tests between all pairs of models and print results in a readable format.
    
    Parameters:
    df (pd.Dataframe): pandas dataframe with 'model' and 'score' columns containing model names and their scores
    modelnames (list): list of model names
    """

    df = df.fillna(0)
    
    print("\nPairwise T-Test Results:")
    print("-" * 100)
    print(f"{'Model A':<30} | {'Model B':<30} | {'t-stat':>8} | {'p-value':>8}")
    print("-" * 100)
    
    # Perform t-test for each pair
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model_a = models[i]
            model_b = models[j]
            modelname_a = modelnames[i].replace('\n','-')
            modelname_b = modelnames[j].replace('\n','-')
            
            # Get scores for each model
            scores_a = df[df['model'] == model_a]['score']
            scores_b = df[df['model'] == model_b]['score']
            
            # Perform t-test
            if not paired:
                t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)  # Using Welch's t-test
            else:
                t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
            
            # Add significance stars
            stars = ''
            if p_value < 0.001:
                stars = '***'
            elif p_value < 0.01:
                stars = '**'
            elif p_value < 0.05:
                stars = '*'
            
            # Print results
            print(f"{modelname_a:<30} | {modelname_b:<30} | {t_stat:>8.3f} | {p_value:>7.3f}{stars}")
    
    print("-" * 100)
    print("Significance levels: * p<0.05, ** p<0.01, *** p<0.001")