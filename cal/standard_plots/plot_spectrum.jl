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
run = DataRun(44)
channel = ChannelId(1)
det = _channel2detector(asic, channel)
category = :cal 
e_type = :e_trap

# load calibrated energy spectrum 
hit_par = Table(read_ldata(asic, :jlhit, category, period, run))
hit_par = hit_par[hit_par.qc]
ecal = getproperty(hit_par, e_type)
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
ecal_config = dataprod_config(asic).energy(filekey).default
µ_literature = round.(ustrip.(ecal_config.co60_lines), digits = 1)
gamma_names = ecal_config.co60_names
fwhm = round.(ustrip.([asic.par[category].rpars.ecal[period, run, channel][e_type].fit[Symbol(line)].fwhm for line in gamma_names]), digits = 2) .* u"keV"
µ = mvalue.(round.(ustrip.([asic.par[category].rpars.ecal[period, run, channel][e_type].fit[Symbol(line)].centroid for line in gamma_names]), digits = 2)) .* u"keV"

# plot 
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1], 
    xticks = 500:500:1500,
        yscale = log10, 
        title = "Calibrated Energy Specrum ($e_type): $period - $run - $det", 
        xlabel = "Energy (keV)",
        ylabel = "Counts",
        titlesize = 16)
hist!(fig[1, 1], ustrip.(ecal), bins = 500:1:1600)
ylims!(ax, 0.5, nothing)
xlims!(ax, 500, nothing)
Makie.text!(ustrip(µ[1])+50, 1000, text = "$(gamma_names[1]): $(µ[1])\nFWHM = $(fwhm[1])"; align = (:right, :center), fontsize = 16)
Makie.text!(ustrip(µ[2])-50, 1000, text = "$(gamma_names[2]): $(µ[2])\nFWHM = $(fwhm[2])"; align = (:left, :center), fontsize = 16)
fig

plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :spectrum) * "/"
if !isdir(plt_folder)
    mkdir(plt_folder)
end
plt_name = plt_folder * _get_pltfilename(asic, filekey, channel, Symbol("spectrum_$(e_type)"))
save(plt_name, fig)

# vlines!(ax, [1450, 1470], color = :red, linestyle = :dash)
# fig

# find waveforms of large peaks 
hit_par.eventnumber[findall( 1450u"keV" .< hit_par.e_trap .< 1470u"keV")]
