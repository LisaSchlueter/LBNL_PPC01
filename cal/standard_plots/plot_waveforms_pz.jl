## analysis of PPC01 data - period 01 (ASIC)
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using HDF5, LegendHDF5IO
using PropDicts
using Unitful
using TypedTables
using CairoMakie, LegendMakie, Makie
using Measures
using LegendDSP
using RadiationDetectorDSP
using IntervalSets
using Measurements: value as mvalue

# inputs and setup 
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = :cal 
asic = LegendData(:ppc01)

# read waveforms (raw tier)
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
data = read_ldata(asic, DataTier(:raw), filekeys, channel)
wvfs = data.waveform
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
Table(data)

# plot waveform with pole-zero correction 
# 1. shift waveforms by baseline and do pole-zero correcition 
bl_stats = signalstats.(wvfs, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
wvfs_shift = shift_waveform.(wvfs, -bl_stats.mean)
τ_pz = mvalue(asic.par[category].rpars.pz[period, run, channel].τ)
deconv_flt = InvCRFilter(τ_pz)
wvfs_pz = deconv_flt.(wvfs_shift)

show_dsp_windows = false 
y_unit = "V"
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :waveform)
begin
    i = rand(1:length(data.waveform))
    fig = Figure(size = (600, 400))
    ax = Axis(fig[1, 1], title = label = "$period-$run - waveform $i", xlabel = "Time ($(unit(data.waveform[i].time[1])))", ylabel = "Signal ($y_unit)")
     if show_dsp_windows
        bw = Makie.vspan!(ax, ustrip(leftendpoint(dsp_config.bl_window)), ustrip(rightendpoint(dsp_config.bl_window)); ymin = 0.0, ymax = 1.0, color = (:lightblue, 0.5), label = false)
        tw = Makie.vspan!(ax, ustrip(leftendpoint(dsp_config.tail_window)), ustrip(rightendpoint(dsp_config.tail_window)); ymin = 0.0, ymax = 1.0, color = (:silver, 0.5), label = false)
        Makie.text!( ustrip(rightendpoint(dsp_config.bl_window))/2, 0.9*maximum(data.waveform[i].signal), text = "baseline"; align = (:center, :center), fontsize = 20, color = :steelblue)
        Makie.text!( ustrip(rightendpoint(dsp_config.bl_window))+ustrip(rightendpoint(dsp_config.tail_window))/2, 0.9*maximum(data.waveform[i].signal), text = "tail"; align = (:center, :center), fontsize = 20, color = :black)
        pname = plt_folder * "/waveform_$(i)_windows.png"
    else
        pname = plt_folder * "/waveform_$(i).png"
    end 
    praw = lines!(ax, ustrip.(wvfs_shift[i].time), wvfs_shift[i].signal , color = :dodgerblue, linewidth = 3, label = "raw")
    ppz =  lines!(ax, ustrip.(wvfs_pz[i].time), wvfs_pz[i].signal , color = :red2, linewidth = 3, label = "pole-zero corrected")
    axislegend(ax, [praw, ppz], ["raw", "pole-zero corrected"], position = :rb, fontsize = 15)
    wvfs_pz
    fig
    save(pname, fig)
    @info "Saved waveform plot to $pname"
    fig
end 