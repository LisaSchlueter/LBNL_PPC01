# activate environment and load modules. 
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendDSP
using LegendDSP: get_fltpars
using LegendSpecFits: get_friedman_diaconis_bin_width
using LegendHDF5IO
using RadiationSpectra
using RadiationDetectorDSP
using Measurements: value as mvalue
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using Plots
using Measures

# set data configuration (where to find data; and where to save results)
if gethostname() == "Lisas-MacBook-Pro.local"
    ENV["LEGEND_DATA_CONFIG"] = "/Users/lisa/Documents/Workspace/LEGEND/LBL_ASIC/ASIC_data/ppc01/config.json"
else # on NERSC 
    # run(`hostname -d`)
    ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"
end 

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/processing_funcs/process_peak_split.jl")
include("$(@__DIR__)/$relPath/src/simple_dsp.jl")
include("$(@__DIR__)/$relPath/src/apply_qc.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# inputs
reprocess = true
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = DataCategory(:cal)
source = :co60

# load configs 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
qc_config = dataprod_config(asic).qc(filekeys[1]).default
ecal_config = dataprod_config(asic).energy(filekeys[1]).default
# ecal_config.co60_lines
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)


# AUTOMATIC PEAK SPLIITING DID NOT WORK CORRECTLY. 2nd peak is strongly suppressed. 
# thats why we do it the manual way, with tuned parameters using only Co60a peak 
mode = :auto 

if mode == :auto
    # run processor 
    plts = process_peak_split(asic, period, run, category, channel, ecal_config, dsp_config, qc_config ; reprocess = reprocess)   

    # plots 
    display(plts[2])
end 


# DEBUG: generate peakfiles manually 
data = asic
filekeys = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])
peak_folder =  data.tier[DataTier(:jlpeaks), category , period, run] * "/"
if !ispath(peak_folder)
    mkpath(peak_folder)
    @info "create path: $peak_folder"
end
peak_files = peak_folder .* string.(filekeys) .* "-tier_jlpeaks.lh5"

if Symbol(ecal_config.source) == :co60
    gamma_lines =  ecal_config.co60_lines
    gamma_names =  ecal_config.co60_names
    left_window_sizes = ecal_config.co60_left_window_sizes 
    right_window_sizes = ecal_config.co60_right_window_sizes 
elseif Symbol(ecal_config.source) == :th228
    gamma_lines =  ecal_config.th228_lines
    gamma_names =  ecal_config.th228_names
    left_window_sizes = ecal_config.left_window_sizes 
    right_window_sizes = ecal_config.right_window_sizes 
end

h_uncals = Vector{Histogram}(undef, length(filekeys))
peakpos = Vector{Vector{<:Real}}(undef, length(filekeys))
f = 5
# for f in eachindex(filekeys) 
    filekey = filekeys[f]
    peak_file = peak_files[f]
    data_ch = read_ldata(data, DataTier(:raw), filekey, channel)
    wvfs = data_ch.waveform
    e_uncal = filter(x -> x >= qc_config.e_trap.min , data_ch.daqenergy)
    if isempty(e_uncal)
        @warn "No energy values >= $(qc_config.e_trap.min) found for $filekey - skip"
        # continue
    end
    eventnumber = data_ch.eventnumber
    timestamp = data_ch.timestamp
   
      # binning and peak search windows and histogram settings 
    bin_min = quantile(e_uncal, ecal_config.left_bin_quantile)
    bin_max = quantile(e_uncal, ecal_config.right_bin_quantile)
    peak_min = quantile(e_uncal,  ecal_config.left_peak_quantile)
    peak_max = quantile(e_uncal, ecal_config.right_peak_quantile)
    bin_width = get_friedman_diaconis_bin_width(filter(in(bin_min..bin_max), e_uncal))
    # ecal_config.nbins_min = 20
    if (peak_max-peak_min)/bin_width < ecal_config.nbins_min
        bin_width = (peak_max-peak_min)/ ecal_config.nbins_min
    end

    # peak search
    h_uncals[f] = fit(Histogram, e_uncal, 0:bin_width:maximum(e_uncal)) # histogram over full energy range; stored for plot 
    h_peaksearch = fit(Histogram, e_uncal, 0:bin_width:peak_max*1.5) # histogram for peak search
    plot(h_peaksearch)
    _, peakpos[f] = RadiationSpectra.peakfinder(h_peaksearch, σ= ecal_config.peakfinder_σ, backgroundRemove=true, threshold = ecal_config.peakfinder_threshold)
    _, peakpos[f] = RadiationSpectra.peakfinder(h_peaksearch, σ= 6.0, backgroundRemove=true, threshold = 10)
  
    plot(h_peaksearch)

    if length(peakpos[f]) !== length(gamma_lines)
        error("Number of peaks found $(length(peakpos[f])); expected gamma lines $(length(gamma_lines)) \n you could try to modify peakfinder_threshold and/or peakfinder_σ")
    else 
        @info "Found $(length(peakpos[f])) peaks for $filekey"
    end 
    cal_simple = mean(gamma_lines./sort(peakpos[f]))
    e_simplecal = e_uncal .* cal_simple


# end 

