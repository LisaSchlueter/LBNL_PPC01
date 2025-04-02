# Purpose: Create peakfiles for a given run and channel.
# 1. load raw data tier (waveforms)
# 2. apply peak finder of DAQ energies (rough energy estimate of each waveform) to locate specified gamma peaks
# 3. perform simple energy calibration
# 4. select waveforms that are within the energy window of the specified gamma peaks
# 5. apply quality cuts
# 6. save waveforms to peakfiles
#  sanity plots document this procedure and are automatically saved in jlplt tier 
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendDSP
using LegendDSP: get_fltpars
using LegendSpecFits: get_friedman_diaconis_bin_width
using LegendHDF5IO
using RadiationSpectra
using RadiationDetectorDSP
using Measurements: value as mvalue
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using Makie, CairoMakie, LegendMakie
using Measures

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/processing_funcs/process_peak_split.jl")
include("$(@__DIR__)/$relPath/src/simple_dsp.jl")
include("$(@__DIR__)/$relPath/src/apply_qc.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# inputs
reprocess = true
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(50)
channel = ChannelId(1)
category = DataCategory(:cal)

# load configs 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
qc_config = dataprod_config(asic).qc(filekeys[1]).default
ecal_config = dataprod_config(asic).energy(filekeys[1]).default
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)

# run processor 
report = process_peak_split(asic, period, run, category, channel, ecal_config, dsp_config, qc_config ; reprocess = reprocess)   

# read peakfiles 
# data_peak  = Table(read_ldata((:Co60a), asic, :jlpeaks, category, period, run, channel))
data_peak  = Table(read_ldata((:Tl208FEP), asic, :jlpeaks, category, period, run, channel))

# plot peak energies (very rough energy estimate; not the final one!!)
fig = Figure()
ax = Axis(fig[1,1], xlabel = "Energy (keV)")
hist!(ax, ustrip.(data_peak.e_simplecal), bins = 50)
fig



