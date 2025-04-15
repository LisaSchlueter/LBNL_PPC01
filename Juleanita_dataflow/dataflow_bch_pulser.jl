# convert .ecsv files from SkuTek "FemtoDAQ Vireo" to LEGEND-style hdf5-files. 
using Juleanita
using LegendDataManagement
using Unitful

# inputs / settings 
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = :bch
timestep = 0.01u"µs"
chmode = :pulser
filter_types = [:trap]

# 0. pre-proccesing: convert csv to lh5 
csv_folder = asic.tier[DataTier(:raw_csv), category , period, run]
skutek_csv_to_lh5(asic, period, run, category, channel, csv_folder; timestep = timestep, chmode = chmode)

# load configs 
filekeys   = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
pz_config  = dataprod_config(asic).dsp(filekeys[1]).pz.default

# 1. decay times for all waveform
plot_τ = process_decaytime(asic, period, run, category, channel, pz_config, dsp_config; reprocess = reprocess)
τ_pz = asic.par[category].rpars.pz[period, run, channel].τ

# 2. filter optimizaiton 
process_filteropt(asic, period, run, category, channel, dsp_config, mvalue(τ_pz), :all; 
                reprocess = true, rt_opt_mode = :bl_noise, filter_types = filter_types)
pars_filter = asic.par[category].rpars.fltopt[period, run, channel]

# 3. run dsp on all waveforms in raw tier. output: dsp files
process_dsp(asic, period, run, category, channel, dsp_config, mvalue(τ_pz), pars_filter; reprocess = reprocess)
dsp_pars = read_ldata(asic, :jldsp, category, period, run, channel);

# 4. fit peaks 
process_peakfits(asic, period, run, category, channel; reprocess = true, juleana_logo = false, rel_cut_fit = 0.1)

# 5. linearity plot 
