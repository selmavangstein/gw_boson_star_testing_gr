import papermill as pm
from pathlib import Path
import scrapbook as sb
import tempfile
import os

# this block is if we want to run one mode for some start times and two modes for others
#num_modes = [1, 2]
#start_offsets2 = [5, 6, 7, 8, 9, 10, 10.5, 11]
#start_offsets1 = [12, 13, 14, 15, 16, 17, 18, 19, 20]
# Define your parameter combinations
# parameter_sets = [
#     {"num_modes": nm, "analysis_start_time": so, "output_file": f"fitresults/fit_result_modes_{nm}_start_{str(so).replace('.', '')}.json"} for nm in num_modes for so in start_offsets2
# ] + [
#     {"num_modes": 1, "analysis_start_time": so, "output_file": f"fitresults/fit_result_modes_1_start_{str(so).replace('.', '')}.json"} for so in start_offsets1
# ]

num_modes = [2]
start_offsets = [12]

parameter_sets = [
    {"num_modes": nm, "analysis_start_time": so, "output_file": f"fitresults/fit_result_modes_{nm}_start_{str(so).replace('.', '')}.json"} for nm in num_modes for so in start_offsets
]


print(f"Total configurations to run: {len(parameter_sets)}")
print("full parameter sets:")
for params in parameter_sets:
    print(params)
# Path to your notebook
notebook_path = "./bs_ringdown.ipynb"



# Run the notebook for each parameter set
for i, params in enumerate(parameter_sets):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as tmp:
        output_notebook = tmp.name
    
    print(f"\nRunning configuration {i+1}/{len(parameter_sets)}")
    print(f"Parameters: {params}")
    
    try:
        pm.execute_notebook(
            notebook_path,
            output_notebook,
            parameters=params,
            kernel_name='ringdown'
        )
        print(f"✓ Completed: {output_notebook}")        
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\nAll runs completed!")