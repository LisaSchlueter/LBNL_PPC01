
# look at high-energy tail at 1460 keV peak (above Co60 spectrum)
using LegendDataManagement
using LegendDataManagement.LDMUtils
using CairoMakie, LegendPlots
using LegendHDF5IO
using PropDicts
using TypedTables
using Statistics, StatsBase
using Unitful, Measures
using Measurements: value as mvalue
using Printf
# inputs
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(2)
channel = ChannelId(1)
category = :cal 
e_type = :e_trap

# load calibrated energy spectrum 
hit_par = Table(read_ldata(asic, :jlhit, category, period, run))
hit_par = hit_par[hit_par.qc]

# find waveforms of large peaks 
idx = findall( 1450u"keV" .< hit_par.e_trap .< 1470u"keV")
eventnumbers = hit_par.eventnumber[idx]
ecal = hit_par.e_trap[idx]

# load waveforms 
data = Table(read_ldata(asic, DataTier(:raw), category, period, run, channel))
wvfs = data.waveform[eventnumbers]

# fig = Figure()
# ax = Axis(fig[1, 1])#, title = "Waveforms of large peaks", xlabel = "Time (ns)", ylabel = "ADC counts")
# i  = 1
# lines!(ax, ustrip.(wvfs[i].time), wvfs[i].signal, label =  @sprintf("%.0f keV", ustrip(ecal[i])), linewidth = 2.5)
# axislegend()
# fig

save_plot = true
ncol = 2
nrow = 5
j = 2
nwaveforms = 10 
nplots = floor(Int, length(wvfs) / nwaveforms)
figs = Vector{Figure}(undef, nplots)
for j = 1:nplots
    nrow = ceil(Int, nwaveforms/ncol)
    figs[j] = Figure(size = (425.0*ncol, nrow*220.0))
    grid = figs[j][1,1] = GridLayout()
   
    for i= 1:nwaveforms
        row = div(i - 1, ncol) + 1
        col = mod(i - 1, ncol) + 1
        ax = Axis(grid[row, col], xlabel = "Time (µs)", ylabel = "Signal (V)")
        wvf_idx = 10*(j-1)+i
        @debug wvf_idx
        lines!(ax, ustrip.(wvfs[wvf_idx].time), ustrip.(wvfs[wvf_idx].signal), linewidth = 2.5, color = :deepskyblue)
        text!(ax, 0.95, 0.05; text = @sprintf("%.0f keV", ustrip(ecal[i])), space = :relative, align = (:right, :bottom), color = :deepskyblue)
    end
    Label(grid[1, 1:2, Top()], "", valign = :bottom, font = :bold, padding = (0, 0, 0, 0), )
    figs[j] 
    if save_plot == true 
        pname = "$(@__DIR__)/plots/" * "waveforms_highEnergyPeak_$(j).png"
        save(pname, figs[j])
    end 
end 
