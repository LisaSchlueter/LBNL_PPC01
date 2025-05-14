env_path = "$(@__DIR__)/" * relpath(split(@__DIR__, "LBNL_PPC01")[1], @__DIR__)  * "/LBNL_PPC01/"
import Pkg; Pkg.activate(env_path)
using LegendDataManagement
using LegendDataManagement: readlprops, writelprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using PropDicts
using Unitful
using TypedTables

# set data configuration (where to find data; and where to save results)
if gethostname() == "Lisas-MacBook-Pro.local"
    ENV["LEGEND_DATA_CONFIG"] = "/Users/lisa/Documents/Workspace/LEGEND/LBL_ASIC/ASIC_data/ppc01/config.json"
  else # on NERSC 
    ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"
  end 
# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/apply_qc.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_qualitycuts.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = :cal 

# load qc config and apply 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
qc_config = dataprod_config(asic).qc(filekeys[1]).default
process_qualitycuts(asic, period, run, category, channel; reprocess = true, qc_config = qc_config);

# read quality cuts from pars 
qc = asic.par.rpars.qc[period][run][channel]

# plot qc 
# --> run plot_qc


