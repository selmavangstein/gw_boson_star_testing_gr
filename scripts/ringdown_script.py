import papermill as pm
from pathlib import Path
import scrapbook as sb
import tempfile
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
RINGDOWN_DIR = PROJECT_DIR.parent / "datadir" / "ringdown_output" / "chaintests"

# this block is if we want to run one mode for some start times and two modes for others
#num_modes = [1, 2]
#start_offsets2 = [5, 6, 7, 8, 9, 10, 10.5, 11]
#start_offsets1 = [12, 13, 14, 15, 16, 17, 18, 19, 20]
# Define your parameter combinations
# parameter_sets = [
#     {"num_modes": nm, "analysis_start_time": so, "output_file": RINGDOWN_DIR / f"fit_result_modes_{nm}_start_{str(so).replace('.', '')}.nc"} for nm in num_modes for so in start_offsets2
# ] + [
#     {"num_modes": 1, "analysis_start_time": so, "output_file": RINGDOWN_DIR / f"fit_result_modes_1_start_{str(so).replace('.', '')}.nc"} for so in start_offsets1
# ]

# parameter_sets = [
#     {"num_modes": nm, "analysis_start_time": so, "output_file": RINGDOWN_DIR / f"fit_result_modes_{nm}_start_{str(so).replace('.', '')}.nc"} for nm in num_modes for so in start_offsets
# ]

num_modes = 2
start_offset = 6

num_chains = [4, 8]
num_samples = [2000, 4000]
num_runs = 2

parameter_sets = [
    {"num_modes": num_modes, 
     "analysis_start_time": start_offset, 
     "num_chains": nc, 
     "num_samples": ns,
     "output_file": RINGDOWN_DIR / f"fit_result_modes_{num_modes}_start_{str(start_offset).replace('.', '')}_chains_{nc}_samples_{ns}_run_{run}.nc"} for nc in num_chains for ns in num_samples for run in range(num_runs)
]





print(f"Total configurations to run: {len(parameter_sets)}")
print("full parameter sets:")
for params in parameter_sets:
    print(params)
# Path to your notebook
notebook_path = PROJECT_DIR / "notebooks" / "bs_ringdown.ipynb"



# Run the notebook for each parameter set
for i, params in enumerate(parameter_sets):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as tmp:
        output_notebook = tmp.name
    
    print(f"\nRunning configuration {i+1}/{len(parameter_sets)}")
    print(f"Parameters: {params}")
    
    try:
        RINGDOWN_DIR.mkdir(parents=True, exist_ok=True)
        pm.execute_notebook(
            str(notebook_path),
            output_notebook,
            parameters=params,
            kernel_name='ringdown'
        )
        print(f"✓ Completed: {output_notebook}")        
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\nAll runs completed!")
