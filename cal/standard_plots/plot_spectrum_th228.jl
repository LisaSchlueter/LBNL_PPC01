using LegendDataManagement
using LegendDataManagement.LDMUtils
using CairoMakie, LegendMakie, Makie
using LegendHDF5IO
using PropDicts
using TypedTables
using Statistics, StatsBase
using Unitful, Measures
using Measurements: value as mvalue

# set data configuration (where to find data; and where to save results)

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(40)
channel = ChannelId(1)
det = _channel2detector(asic, channel)
category = :cal 
e_type = :e_trap
source = :th228

# load calibrated energy spectrum 
hit_par = Table(read_ldata(asic, :jlhit, category, period, run))
hit_par = hit_par[hit_par.qc]
ecal = getproperty(hit_par, e_type)
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
ecal_config = dataprod_config(asic).energy(filekey).default
µ_literature = round.(ustrip.(getproperty(ecal_config, Symbol("$(source)_lines"))), digits = 1)#round.(ustrip.(ecal_config.co60_lines), digits = 1)
gamma_names = getproperty(ecal_config, Symbol("$(source)_names"))
fwhm = round.(ustrip.([asic.par[category].rpars.ecal[period, run, channel][e_type].fit[Symbol(line)].fwhm for line in gamma_names]), digits = 2) .* u"keV"
µ = mvalue.(round.(ustrip.([asic.par[category].rpars.ecal[period, run, channel][e_type].fit[Symbol(line)].centroid for line in gamma_names]), digits = 2)) .* u"keV"

# plot 
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1], 
        xticks = 500:500:3000,
        yscale = log10, 
        title = "Calibrated Energy Specrum ($e_type): $period - $run - $det", 
        xlabel = "Energy (keV)",
        ylabel = "Counts",
        titlesize = 16)
hist!(fig[1, 1], ustrip.(ecal), bins = 500:5:3500)
ylims!(ax, 1, nothing)
xlims!(ax, 500, 3000)

for peak in round.(Int, µ_literature)
    if peak == 1592
        Makie.text!(peak - 25, 6000, text = "$peak"; align = (:center, :center), fontsize = 16, rotation = pi/2, color = :darkgrey)
    elseif peak == 1620
        Makie.text!(peak + 25, 6000, text = "$peak"; align = (:center, :center), fontsize = 16, rotation = pi/2, color = :darkgrey)
    else
        Makie.text!(peak, 6000, text = "$peak"; align = (:center, :center), fontsize = 16, rotation = pi/2, color = :darkgrey)
    end 
end 
fig

plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :spectrum) * "/"
if !isdir(plt_folder)
    mkdir(plt_folder)
end
plt_name = plt_folder * _get_pltfilename(asic, filekey, channel, Symbol("spectrum_$(e_type)"))
save(plt_name, fig)
fig

