import argparse
import importlib.util
import os


def load_python_file(filepath):
    filepath = os.path.abspath(filepath)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find python file: {filepath}")

    module_name = os.path.splitext(os.path.basename(filepath))[0]

    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def main(paper, figure, output_dir):
    paper_file = f"src/illustrations/{paper}.py"
    try:
        paper_module = load_python_file(paper_file)
    except Exception as e:
        print(f"Error loading file {paper_file}: {e}")
        exit(1)

    func_name = f"plot_fig_{figure}"
    if not hasattr(paper_module, func_name):
        print(f"Error: function '{func_name}' not found in {paper_file}")
        exit(1)

    plot_func = getattr(paper_module, func_name)

    if output_dir is not None:
        output_file = os.path.join(output_dir, f"{paper}-Fig{figure}.png")
    else:
        output_file = None

    plot_func(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce plots from papers.")
    parser.add_argument(
        "--paper",
        "-p",
        type=str,
        required=True,
        help="Which paper to reproduce figures from. (format: NameYear)",
    )
    parser.add_argument(
        "--figure",
        "-f",
        type=str,
        required=True,
        help="Which figure to generate.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./outputs",
        help="Output directory to save figure. If not provided, the plot is shown.",
    )
    # parser.add_argument(
    #     "--kwargs",
    #     type=str,
    #     default=None,
    #     help=(
    #         "Optional additional keyword arguments for the plotting function. "
    #         'Pass as a dictionary string, e.g., \'{"param1": 10, "param2": "value"}\''
    #     ),
    # )
    args = parser.parse_args()

    # Parse additional kwargs
    # import ast
    # extra_kwargs = {}
    # if args.kwargs:
    #     try:
    #         extra_kwargs = ast.literal_eval(args.kwargs)
    #         if not isinstance(extra_kwargs, dict):
    #             raise ValueError()
    #     except Exception:
    #         print(
    #             "Error: --kwargs must be a dictionary string, e.g., '{\"param\": 10}'"
    #         )
    #         exit(1)

    main(args.paper, args.figure, args.output_dir)
