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

# load functions from hpge-ana

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

# plot waveforms 
y_unit = "V"
show_dsp_windows = true 
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :waveform)
begin
    i = rand(1:length(wvfs))
    fig = Figure(size = (600, 400))
    ax = Axis(fig[1, 1], title = "$period-$run - waveform $i", xlabel = "Time ($(unit(wvfs[i].time[1])))", ylabel = "Signal ($y_unit)")
     if show_dsp_windows
        bw = Makie.vspan!(ax, ustrip(leftendpoint(dsp_config.bl_window)), ustrip(rightendpoint(dsp_config.bl_window)); ymin = 0.0, ymax = 1.0, color = (:lightblue, 0.5), label = "baseline window")
        tw = Makie.vspan!(ax, ustrip(leftendpoint(dsp_config.tail_window)), ustrip(rightendpoint(dsp_config.tail_window)); ymin = 0.0, ymax = 1.0, color = (:silver, 0.5), label = "tail window")
        p = lines!(ax, ustrip.(wvfs[i].time), wvfs[i].signal , color = :dodgerblue, linewidth = 3, label = "waveform $i")
        Makie.text!( ustrip(rightendpoint(dsp_config.bl_window))/2, 0.9*maximum(wvfs[i].signal), text = "baseline"; align = (:center, :center), fontsize = 20, color = :steelblue)
        Makie.text!( ustrip(rightendpoint(dsp_config.bl_window))+ustrip(rightendpoint(dsp_config.tail_window))/2, 0.9*maximum(data.waveform[i].signal), text = "tail"; align = (:center, :center), fontsize = 20, color = :black)
     pname = plt_folder * "/waveform_$(i)_windows.png"
    else
        pname = plt_folder * "/waveform_$(i).png"
    end 
    p = lines!(ax, ustrip.(wvfs[i].time), wvfs[i].signal , color = :dodgerblue, linewidth = 3, label = "waveform $i")
    fig
    save(pname, fig)
    @info "Saved waveform plot to $pname"
    fig
end 
