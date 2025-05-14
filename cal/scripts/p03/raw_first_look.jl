using LegendDataManagement
using LegendHDF5IO
using Makie, CairoMakie, LegendMakie

asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(52)
channel = ChannelId(1)
category = DataCategory(:cal)

# plot raw data 
e  = filter(x-> 200 < x < 5000, read_ldata(:daqenergy, asic, DataTier(:raw), category, period, run, channel))
fig = Figure()
ax = Axis(fig[1, 1], limits = ((nothing, nothing), (0, nothing)), xlabel = "Pulse height (ADC)", ylabel = "Counts / bin")
hist!(ax, e, bins = 100, color = :blue)
fig 

# guess calibration and plot roughly calibrated spectrum 
fep_adc = 4400
vlines!(ax,[fep_adc], color = :red)
fig
c = 2600 ./ fep_adc 
fig_cal = Figure()
ax = Axis(fig_cal[1, 1], limits = ((nothing, 4000), (0, nothing)), xlabel = "Pulse height (keV)", ylabel = "Counts / bin")
Makie.hist!(ax, e .* c, bins = 100, color = :blue, label = "Th228")
fig_cal

gamma_lines = [583.191,  727.330,  860.564,  1592.53,    1620.50,    2103.53,    2614.51]
for line in gamma_lines
    vlines!(ax, [line], label = "$line keV", linewidth = 2.0, alpha = 0.8, linestyle = :dash)
end
axislegend()
fig_cal