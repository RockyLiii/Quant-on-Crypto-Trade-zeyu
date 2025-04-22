# Crypto Quant

A statistical analysis and visualization tool for cryptocurrency market data, focusing on correlation analysis, residual calculations, and trading signal generation.

## Project Structure

```
StatAgent/
├── config/             # Configuration files (YAML)
├── data/              # Input data directory
├── legacy/            # Legacy code and files
├── output/            # Generated output files and visualizations
├── script/            # Utility scripts
├── src/               # Source code
│   ├── dataloader/    # Data loading utilities
│   ├── utils/         # Calculation utilities
│   └── visual/        # Visualization modules
├── main.py            # Main entry point
└── requirement.txt    # Project dependencies
```

## Usage

Run the main script with a configuration file:

```bash
python main.py
```

The program will:
1. Load cryptocurrency data from the specified directory
2. Perform statistical analysis
3. Generate visualizations
4. Save results to the output directory

## Configuration

The project uses YAML configuration files to control:
- Input data directory and file patterns
- Analysis parameters
- Visualization settings
- Output paths

Example configuration files can be found in the `config/` directory.

## Output

Results are saved in the `output/` directory, including:
- Log files
- Generated visualizations
