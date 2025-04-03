using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using StatsBase
using LegendSpecFits
using Makie, LegendMakie, CairoMakie   
using PropDicts
using TypedTables
# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_peakfits.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(1)
channel = ChannelId(1)
category = DataCategory(:bch)
run = DataRun(7)
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])

# read dsp pars and fit peaks 
dsp_pars = read_ldata(asic, :jldsp, category, period, run, channel);
process_peakfits(asic, period, run, category, channel; reprocess = true, juleana_logo = false, rel_cut_fit = 0.1)

