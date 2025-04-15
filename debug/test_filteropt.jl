


using LegendSpecFits
using LegendDSP
using RadiationDetectorDSP
using RadiationDetectorSignals
using LegendHDF5IO, HDF5
using Unitful
using PropDicts
# create baseline waveform 
url = "https://github.com/legend-exp/legend-testdata/blob/main/data/lh5/prod-ref-l200/generated/tier/raw/cal/p03/r001/l200-p03-r001-cal-20230318T012144Z-tier_raw.lh5"

# generate step-form like waveforms  (randomized)
n = 600
reltol = 1e-6
t = range(0u"μs", 20u"μs", 2*n)
nsamples = 1000
signal = vcat(zeros(n), 10*ones(n)) 
signal = repeat(signal', nsamples)' .+ 0.05 .* randn(length(signal), nsamples)
wf = ArrayOfRDWaveforms([RDWaveform(t, signal[:,i]) for i = 1:nsamples])

filter_type = :trap
rt = (0.5:0.5:20.0).*u"µs"
def_ft = 0.1u"µs"
dsp_config_mod = DSPConfig(PropDict( 
                    "e_grid_$(filter_type)" => PropDict("rt" => PropDict(:start => minimum(rt), :stop => maximum(rt), :step => step(rt))),
                    "flt_defaults" => PropDict("$(filter_type)" => PropDict(:ft => def_ft, :rt => 1.0u"µs")),
                    "flt_length_cusp" =>  1.0u"µs",
                    "flt_length_cusp" =>  1.0u"µs"))

