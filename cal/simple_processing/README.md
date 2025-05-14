# `simple_processing.jl`

This script provides a quick and simple way to go from raw waveforms to a calibrated spectrum **without running the full processing pipeline**.  
It does not rely on configuration files — everything is handled directly within the script itself, making it highly configurable and easy to tweak.

> ⚠️ **Note:** The results from this script are **preliminary** and not optimized. It is intended to give a **first impression** of the data, not a final analysis.

## ✅ Use Cases

- When you want a **quick look at the data** without setting up full configs
- For **unusual data** (e.g. from a GFET amplifier) that might need custom treatment
- To **help select configuration options** for later full processing
