using Juleanita
using LegendDataManagement
using LegendDataManagement.LDMUtils
using LegendDSP: DSPConfig
using Unitful
using PropDicts
using Printf
using Makie, LegendMakie, CairoMakie

# inputs / settings 
asic = LegendData(:ppc01)
period = DataPeriod(1)

run = DataRun(1)
channel = ChannelId(1)
category = :bch
timestep = 0.01u"µs"
chmode = :single
reprocess = false
# 0. pre-proccesing: convert csv to lh5 
csv_folder = asic.tier[DataTier(:raw_csv), category , period, run]
skutek_csv_to_lh5(asic, period, run, category, channel, csv_folder; timestep = timestep, chmode = chmode)

# load configs and modify if needed 
waveform_type = :waveform 
n_evts = 2000
filter_type = :trap
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
rt = (1.0:0.5:40.5).*u"µs"
def_ft = 0.5u"µs"
pd_default = dataprod_config(asic).dsp(filekeys[1]).default
dsp_config_mod = DSPConfig(merge(pd_default, PropDict( 
                    "e_grid_$(filter_type)" => PropDict("rt" => PropDict(:start => minimum(rt), :stop => maximum(rt), :step => step(rt))),
                    "flt_defaults" => PropDict("$(filter_type)" => PropDict(:ft => def_ft, :rt => 1.0u"µs")),
                    "flt_length_cusp" =>  1.0u"µs",
                    "flt_length_cusp" =>  1.0u"µs")))

result_noise, report_noise = process_noisesweep(asic, period, run, category, channel, dsp_config_mod; 
reprocess = reprocess, filter_type = filter_type, waveform_type = :waveform, n_evts = n_evts,
diff_output = false);
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
asic_meta = asic.metadata.hardware.asic(filekeys[1])

plt = plot_noise_sweep(report_noise, :e; 
                DAQ_bits = 14, 
                DAQ_dynamicrange_V  = 2.0, 
                gain = asic_meta.gain_tot, 
                cap_inj = ustrip.(uconvert(u"F", asic_meta.cap_inj)),
                title = get_plottitle(filekeys[1], _channel2detector(asic, channel), "Noise sweep") * @sprintf("\n gain add. = %.1f, Cf = %.0f fF, Cinj = %.0f fF", asic_meta.gain_tot, ustrip.(uconvert(u"fF", asic_meta.cap_feedback)), ustrip.(uconvert(u"fF", asic_meta.cap_inj)) )).fig 

