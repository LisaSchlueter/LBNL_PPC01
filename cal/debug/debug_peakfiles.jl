
# Debug script for peakfiles. 
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
using LegendMakie, Makie, CairoMakie
using Measures

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
period = DataPeriod(3)
run = DataRun(50)
channel = ChannelId(1)
category = DataCategory(:cal)
source = :th228

# load configs 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
qc_config = dataprod_config(asic).qc(filekeys[1]).default
ecal_config = dataprod_config(asic).energy(filekeys[1]).default
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)

# DEBUG
data = asic
filekeys = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])
peakfinder_gamma = 1
if Symbol(ecal_config.source) == :co60
    gamma_lines =  ecal_config.co60_lines
    gamma_names =  ecal_config.co60_names
    left_window_sizes = ecal_config.co60_left_window_sizes 
    right_window_sizes = ecal_config.co60_right_window_sizes 
elseif Symbol(ecal_config.source) == :th228
    gamma_lines =  ecal_config.th228_lines[end]
    gamma_names =  [ecal_config.th228_names[end]]
    left_window_sizes = 1.5 * ecal_config.th228_left_window_sizes[end]
    right_window_sizes = 1.5 * ecal_config.th228_right_window_sizes[end] 
end

if peakfinder_gamma !== NaN
    gamma_lines =  gamma_lines[peakfinder_gamma]
    gamma_names =  gamma_names[peakfinder_gamma]
    left_window_sizes = left_window_sizes[peakfinder_gamma]
    right_window_sizes = right_window_sizes[peakfinder_gamma]
    @info "use only peakfinder_gamma $(peakfinder_gamma) for peak search"
end
# for f in eachindex(filekeys) 
# filekey = filekeys[f]
# peak_file = peak_files[f]
data_ch = read_ldata(data, DataTier(:raw), filekeys, channel)
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
peak_min = quantile(e_uncal, ecal_config.left_peak_quantile)
peak_max = quantile(e_uncal, ecal_config.right_peak_quantile)
bin_width = LegendSpecFits.get_friedman_diaconis_bin_width(filter(in(bin_min..bin_max), e_uncal))
if (peak_max-peak_min)/bin_width < ecal_config.nbins_min
    bin_width = (peak_max-peak_min)/ ecal_config.nbins_min
end

# peak search
# h_uncal = fit(Histogram, e_uncal, 0:bin_width:maximum(e_uncal)) # histogram over full energy range; stored for plot 

peakpos = []
function _peakfinder(p_min::T, p_max::T) where T <: Real
    try 
        h_peaksearch = fit(Histogram, e_uncal, p_min:bin_width:p_max) # histogram for peak search
        _, peakpos = RadiationSpectra.peakfinder(h_peaksearch, σ= ecal_config.peakfinder_σ, backgroundRemove=true, threshold = ecal_config.peakfinder_threshold)
        @info "Found $(length(peakpos)) peak(s) at $(peakpos)"
        return peakpos
    catch e
        @info "peakfinder failed: $e"
        return NaN
    end
end

peakpos = _peakfinder(peak_min, peak_max)
if peakpos == NaN
    @info "try larger peak window"
    peakpos = _peakfinder(peak_min, 1.5*peak_max)  
elseif length(peakpos) > length(gamma_lines) 
    @info "too many peaks found - try smaller peak window"
    for i in 0.1:0.1:0.9
        peakpos = _peakfinder((1+i)*peak_min, (1-i)*peak_max)
        if length(peakpos) == length(gamma_lines) 
            break 
        end 
    end 
elseif length(peakpos) < length(gamma_lines)
  @info "too few peaks found - try larger peak window"
    for i in 0.1:0.1:0.9
        peakpos = _peakfinder((1-i)*peak_min, (1+i)*peak_max)
        if length(peakpos) == length(gamma_lines) 
            break 
        end 
    end 
end 



# try
#     h_peaksearch = fit(Histogram, e_uncal, bin_min:bin_width:peak_max) # histogram for peak search
#     _, peakpos[f] = RadiationSpectra.peakfinder(h_peaksearch, σ= ecal_config.peakfinder_σ, backgroundRemove=true, threshold = ecal_config.peakfinder_threshold)
# catch e 
#     @warn "peakfinder failed for $filekey - use larger window. err $e"
#     h_peaksearch = fit(Histogram, e_uncal, bin_min:bin_width:(peak_max*1.5)) # histogram for peak search
#     _, peakpos[f] = RadiationSpectra.peakfinder(h_peaksearch, σ= ecal_config.peakfinder_σ, backgroundRemove=true, threshold = ecal_config.peakfinder_threshold)
# end 
# if length(peakpos[f]) !== length(gamma_lines)
#     error("Number of peaks found $(length(peakpos[f])); expected gamma lines $(length(gamma_lines)) \n you could try to modify peakfinder_threshold and/or peakfinder_σ (file $f)")
# else 
#     @info "Found $(length(peakpos[f])) peaks for $filekey"
# end 
cal_simple = mean(gamma_lines./sort(peakpos))
e_simplecal = e_uncal .* cal_simple
i = 1
peakIdx = findall((gamma_lines[i] - left_window_sizes[i]) .<= e_simplecal .< (gamma_lines[i] + right_window_sizes[i]))
# do simple dsp. only for peaks. 
dsp_par = simple_dsp_qc(Table(waveform = wvfs[peakIdx]), dsp_config)

