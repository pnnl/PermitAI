

def main():
    # Argument parser
    import argparse
    from omegaconf import OmegaConf
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--configPath",
        help = "path to the configuration file", 
        default = "configs/v1-eis-mistral.yaml"
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.configPath)

    # Logger setup
    from utils.logging_utils import LogManager
    LogManager.initialize(conf["log_file"])
    logger = LogManager.get_logger("generate_plots")
    logger.info("Plotting application started")

    # Metric to plot
    metric_to_plot = conf["metric"]
    path_pfx = conf["path_pfx"]
    in_percent = conf["percentage"]

    # Load the module classes/methods
    from utils.plot_utils import HeatmapGenerator, BarPlotGenerator

    # Multi-context heatmap generator
    if "multi_context" in conf:
        contexts = conf["multi_context"]["context_type"]
        models = conf["multi_context"]["models"]

        # plot arguments
        width = conf["multi_context"]["plot_args"]["width"]
        height = conf["multi_context"]["plot_args"]["height"]
        xtickrotation = conf["multi_context"]["plot_args"]["xtickrotation"]
        labelsize = conf["multi_context"]["plot_args"]["labelsize"]
        ticklabelsize = conf["multi_context"]["plot_args"]["ticklabelsize"]
        annotation_size = conf["multi_context"]["plot_args"]["annotation_size"]
        colormap = conf["multi_context"]["plot_args"]["colormap"]

        for model,model_name in models.items():
            HeatmapGenerator.generate_matrix_heatmap_multiple_context(
                path_pfx = path_pfx, 
                model = model,
                model_name = model_name, 
                context = contexts, 
                metric = metric_to_plot, 
                output_path = f"plots/{metric_to_plot}_{model}.png",
                figsize=(width, height),
                xtickrotation=xtickrotation,
                labelsize=labelsize,
                ticklabelsize=ticklabelsize,
                annotation_size=annotation_size,
                colormap=colormap,
                percentage=in_percent
                )
    logger.info("Heatmap generation complete for LLMs with multi-context.")

    if "single_context" in conf:
        context = conf["single_context"]["context_type"]
        models,model_names = zip(*conf["single_context"]["models"].items())
        width = conf["single_context"]["plot_args"]["width"]
        height = conf["single_context"]["plot_args"]["height"]
        xtickrotation = conf["single_context"]["plot_args"]["xtickrotation"]
        labelsize = conf["single_context"]["plot_args"]["labelsize"]
        ticklabelsize = conf["single_context"]["plot_args"]["ticklabelsize"]
        annotation_size = conf["single_context"]["plot_args"]["annotation_size"]
        colormap = conf["single_context"]["plot_args"]["colormap"]
        
        # Single context heatmap generator
        HeatmapGenerator.generate_matrix_heatmap_single_context(
            path_pfx = path_pfx, 
            models = models, 
            metric = metric_to_plot, 
            output_path = conf["single_context"]["outputs"]["heatmap"], 
            context_type=context,
            xticklabels=model_names,
            figsize=(width, height),
            xtickrotation=xtickrotation,
            labelsize=labelsize,
            ticklabelsize=ticklabelsize,
            annotation_size=annotation_size,
            colormap=colormap,
            percentage=in_percent
            )

        # Single context barplot generator
        BarPlotGenerator.generate_bar_plot(
            path_pfx = path_pfx, 
            models = models, 
            metric = metric_to_plot, 
            output_path = conf["single_context"]["outputs"]["barplot"], 
            context_type=context,
            xticklabels=model_names,
            figsize=(width, height),
            xtickrotation=xtickrotation,
            labelsize=labelsize,
            ticklabelsize=ticklabelsize,
            annotation_size=annotation_size,
            percentage=in_percent
        )

    
    return


# Example usage
if __name__ == "__main__":
    main()