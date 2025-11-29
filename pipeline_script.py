import papermill as pm
from pathlib import Path
import scrapbook as sb
import tempfile
import os

luminosity_distances = [160, 170, 180, 190, 200]

#possible parameters to vary:
# outdir, filepath, 
# luminosity_distance, mototal,
# beginning_taper_safety, end_taper_safety, pad_factor
# inclination, phiRef, psi, ra, dec
# (these are t/f) bandpass_filter, window_filter, taper_end, dense_star, plotting

# Define your parameter combinations
parameter_sets = [
    {"luminosity_distance": d} for d in luminosity_distances
]

# Path to your notebook
notebook_path = "full_working_pipeline.ipynb"

results = []
# if the folder for executed notebooks doesn't exist, create it


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
        nb = sb.read_notebook(str(output_notebook))
        snr = nb.scraps["h1_snr"].data
        luminosity_distance = nb.scraps["luminosity_distance"].data
        
        results.append({"luminosity_distance": luminosity_distance, "snr": snr})
        print(f"✓ luminosity_distance={luminosity_distance}, snr={snr}")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\nAll runs completed!")
print("====Results=====:")
for result in results:
    print(f"luminosity_distance={result['luminosity_distance']}, snr={result['snr']}")