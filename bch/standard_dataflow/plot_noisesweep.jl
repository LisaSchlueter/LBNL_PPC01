using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using LegendSpecFits
using LegendDSP
using LegendDSP: get_fltpars
using RadiationDetectorSignals
using RadiationDetectorDSP
using Measurements: value as mvalue
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using Makie, LegendMakie, CairoMakie
using Measures
using Optim
using BSplineKit
using Printf

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/enc_noisesweep.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_noisesweep.jl")
include("$(@__DIR__)/$relPath/utils/utils_physics.jl")
# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(11)
channel = ChannelId(1)
category = DataCategory(:bch)
filter_types = [:trap]

# load configs and modify if needed 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1:4]
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
#pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default
peak =  :all
#τ_pz = mvalue(get_values(asic.par.rpars.pz[period, run, channel]).τ)

# do noisesweep

process_noisesweep(asic, period, run, category, channel, dsp_config, peak; 
                reprocess = true, filter_types = filter_types)
# read noisesweep pars
noisesweep_pars = asic.par.rpars.noisesweep[period,run,channel]
