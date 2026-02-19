#!/usr/bin/env python
"""
Command-line interface for running the MQC pipeline
in a containerized environment. It accepts input files/directories and
configuration, then executes the batch processing pipeline.

Examples:
  # Using config file (config file contains input and output paths)
  python run.py --config /data/config/config.yaml[json]

  # Using direct input/output paths
  python run.py --input /data/input/smiles.txt --output /data/output

  # With XYZ directory input
  python run.py --input /data/input/xyz_dir --output /data/output
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import mqc_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from mqc_pipeline.settings import ContainerPipelineSettings
from mqc_pipeline.workflow.pipeline import run_one_batch
from mqc_pipeline.workflow.io import read_smiles, read_xyz_dir
from mqc_pipeline.util import get_default_logger, change_dir
from mqc_pipeline.validate import validate_input

logger = get_default_logger()

_DEFAULT_OUTPUT_DIR = "/data/output"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Docker entrypoint for MQC Pipeline batch processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    parser.add_argument(
        "--config",
        type=str,
        metavar="CONFIG_FILE",
        help=
        "Path to configuration file in YAML or JSON format. If provided, other arguments override config values."
    )

    parser.add_argument(
        "--input",
        type=str,
        metavar="INPUT_PATH",
        help=
        "Path to input file (SMILES .txt/.csv/.pkl) or directory (XYZ files). "
        "Required if --config is not provided.")

    parser.add_argument(
        "--output",
        type=str,
        metavar="OUTPUT_DIR",
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {_DEFAULT_OUTPUT_DIR})")

    parser.add_argument("--write-default-config",
                        type=str,
                        metavar="YAML_FILE",
                        help="Write a default configuration file and exit.")

    return parser.parse_args()


def load_settings(config_path: str | None = None,
                  input_path: str | None = None,
                  output_dir: str | None = None) -> ContainerPipelineSettings:
    settings_dict = {}

    # Load from config file if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        logger.info(f"Loading configuration from {config_path}")
        settings = ContainerPipelineSettings.from_file(config_path)
        settings_dict = settings.model_dump()
    else:
        settings_dict = {}

    # Override with CLI arguments if provided
    if input_path:
        settings_dict['input_file_or_dir'] = str(input_path)
    if output_dir:
        settings_dict['output_dir'] = str(output_dir)

    if 'input_file_or_dir' not in settings_dict:
        raise ValueError(
            "Input path must be provided via --input or in config file")

    try:
        settings = ContainerPipelineSettings(**settings_dict)
    except Exception as e:
        logger.error(f"Failed to create settings: {e}")
        raise

    return settings


def validate_input_path(input_path: str) -> None:
    """
    Validate that input file or directory exists and is readable.

    :param input_path: Path to input file or directory.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}. "
            "Ensure the input is mounted as a volume in the container.")

    if input_path.is_file():
        valid_extensions = {
            '.txt', '.csv', '.pkl', '.pickle', '.parquet', '.parq'
        }
        if input_path.suffix not in valid_extensions:
            logger.warning(
                f"Input file extension '{input_path.suffix}' may not be supported. "
                f"Supported: {', '.join(valid_extensions)}")
    elif input_path.is_dir():
        # Check if directory contains XYZ files
        xyz_files = list(input_path.glob('*.xyz'))
        if len(xyz_files) == 0:
            logger.warning(f"No .xyz files found in directory: {input_path}")
    else:
        raise ValueError(
            f"Input path is neither a file nor directory: {input_path}")


def read_inputs(input_path: str, pickled_input_type: str | None = None):
    """
    Read inputs from file or directory.

    :param input_path: Path to input file or directory.
    :param pickled_input_type: Type of pickled input ('smiles' or 'structure').
    :return: List of SMILES strings or Structure objects.
    """
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix in {'.pkl', '.pickle'}:
            # Handle pickled inputs
            import pickle
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            if pickled_input_type == 'structure':
                # Return list of Structure objects
                return data if isinstance(data, list) else [data]
            else:
                # Return list of SMILES strings
                return data if isinstance(data, list) else [data]
        else:
            # Read SMILES from text/csv/parquet file
            return read_smiles(str(input_path))
    else:
        # Read XYZ files from directory
        return list(read_xyz_dir(str(input_path)))


def main():
    """Main entry point."""
    args = parse_args()

    if args.write_default_config:
        output_path = Path(args.write_default_config)
        ContainerPipelineSettings.write_default_config_to_yaml(output_path)
        logger.info(f"Default configuration written to {output_path}")
        return 0

    try:
        settings = load_settings(config_path=args.config,
                                 input_path=args.input,
                                 output_dir=args.output)

        validate_input_path(settings.input_file_or_dir)

        try:
            validate_input(settings.input_file_or_dir,
                           settings.pickled_input_type)
        except Exception as e:
            logger.warning(f"Input validation warning: {e}")

        # Ensure output directory exists
        output_dir = Path(settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Read inputs
        logger.info(f"Reading inputs from: {settings.input_file_or_dir}")
        inputs = read_inputs(settings.input_file_or_dir,
                             settings.pickled_input_type)

        if len(inputs) == 0:
            logger.error("No inputs found. Exiting.")
            return 1

        logger.info(f"Loaded {len(inputs)} input molecules")

        # Change to output directory for processing
        # run_one_batch writes files to current working directory
        with change_dir(str(output_dir)):
            run_one_batch(inputs, settings)

        logger.info("Pipeline completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
