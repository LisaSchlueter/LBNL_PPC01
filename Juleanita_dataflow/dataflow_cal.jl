using Juleanita
using LegendDataManagement
using LegendDataManagement: LDMUtils
using LegendDSP: DSPConfig 
using Measurements

# inputs
reprocess = true
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = DataCategory(:cal)
filter_types = [:trap]
e_types = Symbol.("e_" .* String.(filter_types))

# load configs 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
qc_config = dataprod_config(asic).qc(filekeys[1]).default
ecal_config = dataprod_config(asic).energy(filekeys[1]).default
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default

# 0. pre-processing: conversion from csv to lh5 file format 
# csv_folder = asic.tier[DataTier(:raw_csv), category , period, run] * "/"
# timestemp = 0.01u"µs"
# skutek_csv_to_lh5(asic, period, run, category, channel, csv_folder; chmode = :diff)

# run processors: 
# 1. create peak files based on rough energy estimate (max-min). output: peak files, and plots
report_ps = process_peak_split(asic, period, run, category, channel, ecal_config, dsp_config, qc_config ; reprocess = reprocess)   

# 2. calculate decay time of all waveforms in peakfile. fit decay time distribution and save for pz-correction
plot_τ    = process_decaytime(asic, period, run, category, channel, pz_config, dsp_config; reprocess = reprocess)
τ_pz = asic.par[category].rpars.pz[period, run, channel].τ

# 3. filteroptimization based on peakfiles: output: rpars.fltopt and plots
process_filteropt(asic, period, run, category, channel, dsp_config, mvalue(τ_pz), Symbol(pz_config.peak); reprocess = reprocess, rt_opt_mode = :bl_noise, filter_types = filter_types)
pars_filter = asic.par[category].rpars.fltopt[period, run, channel]

# 4. run dsp on all waveforms in raw tier. output: dsp files
process_dsp(asic, period, run, category, channel, dsp_config, mvalue(τ_pz), pars_filter; reprocess = reprocess)
dsp_pars = read_ldata(asic, :jldsp, category, period, run, channel);

# 5. apply pre-defined quality cuts based on dsp parameters. output: rpars.qc and qc flag in dsp tier
process_qualitycuts(asic, period, run, category, channel; reprocess = true, qc_config = qc_config);
qc_pars = asic.par[category].rpars.qc[period, run, channel]
dsp_pars.qc

# 6. energy calibration. output: rpars.ecal and plots
process_energy_calibration(asic, period, run, category, channel, ecal_config; reprocess = reprocess, e_types = e_types, plot_residuals = :abs)

# 7. apply energy calibration to all waveforms in dsp tier. output: hit files
process_hit(asic, period, run, category, channel; reprocess = reprocess, e_types = e_types)

