
env_path = "$(@__DIR__)/" * relpath(split(@__DIR__, "LBNL_PPC01")[1], @__DIR__)  * "/LBNL_PPC01/"
import Pkg; Pkg.activate(env_path)
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using LegendSpecFits
using LegendDSP
using LegendDSP: get_fltpars
# using RadiationDetectorSignals
# using RadiationDetectorDSP
using Measurements: value as mvalue
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using Plots
using Measures
using Printf

# set data configuration (where to find data; and where to save results)
ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"

# include relevant functions 
include("$(@__DIR__)/$relPath/src/simple_dsp.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_dsp.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:cal)

# debug
data = asic 

# load config 
dsp_config = DSPConfig(data.metadata.config.dsp.dsp_config.default)
τ_pz = mvalue(data.par.rpars.pz[period, run, channel].τ)
pars_filter = data.par.rpars.fltopt[DataPeriod(1),run,channel] # take optimized filter parameters from period 1

# do dsp
process_dsp(asic, period, run, category, channel; 
            reprocess = true,
            dsp_config = dsp_config,
            τ_pz = τ_pz,
            pars_filter = pars_filter)


# read dsp pars
dsp_pars = read_ldata(asic, :jldsp, category, period, run, channel);
Table(dsp_pars)
columnnames(Table(dsp_pars))

