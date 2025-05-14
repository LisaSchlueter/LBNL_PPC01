## analysis of PPC01 data - period 02 (GFET)
# activate environment and load modules. 
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using HDF5, LegendHDF5IO
using PropDicts
using Unitful
using TypedTables
using CairoMakie 
using Measures
using Measurements: value as mvalue

# set data configuration (where to find data; and where to save results)
if gethostname() == "Lisas-MacBook-Pro.local"
    ENV["LEGEND_DATA_CONFIG"] = "/Users/lisa/Documents/Workspace/LEGEND/LBL_ASIC/ASIC_data/ppc01/config.json"
  else # on NERSC 
    ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"
  end 
  
# load functions from hpge-ana
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__)
include("$relPath/hpge-ana/utils/utils_plot.jl")

# inputs and setup 
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = :cal 
asic = LegendData(:ppc01)

# read waveforms (raw tier)
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
data = read_ldata(asic, DataTier(:raw), filekeys, channel)
Table(data)

wvfs = data.waveform
dsp_config = DSPConfig(asic.metadata.config.dsp.dsp_config.default)

# shift waveforms  by baseline mean 
bl_stats = signalstats.(wvfs, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
wvfs = shift_waveform.(wvfs, -bl_stats.mean)

# do pole-zero correcition 
τ_pz = mvalue(asic.par.rpars.pz[period, run, channel].τ)
deconv_flt = InvCRFilter(τ_pz)
wvfs_pz = deconv_flt.(wvfs)

# plot waveforms with and w/o pole-zero correction
Makie_theme(; fs = 22, xgridvisible = true, ygridvisible = true)
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :waveform_pz)
begin
    i = rand(1:length(data.waveform))
    fig = Figure(size = (600, 400), resolution = (600, 400))
    ax = Axis(fig[1, 1], title = label = "Waveform $i", xlabel = "Time ($(unit(data.waveform[i].time[1])))", ylabel = "Signal (V)")
    p = lines!(ax, wvfs[i].time, wvfs[i].signal , color = :dodgerblue, linewidth = 3, label = "raw")
    lines!(ax, wvfs_pz[i].time, wvfs_pz[i].signal , color = :red2, linewidth = 3, label = "pole-zero corrected")
    axislegend(ax, position = :rb)
    fig
    pname = plt_folder * "/waveform_pz_$(i).png"
    save(pname, fig)
    @info "Saved waveform plot to $pname"
    fig
end 





