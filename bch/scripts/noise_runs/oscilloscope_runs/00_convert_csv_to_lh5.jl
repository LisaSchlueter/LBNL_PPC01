using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using CSV, DataFrames
using HDF5, LegendHDF5IO
using Dates
using PropDicts
using IntervalSets
using Unitful
using TypedTables
using RadiationDetectorDSP
using RadiationDetectorSignals

# include relevant functions
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_IO.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = :bch
ti = 0.0u"µs"..4000.0u"µs" # time interval within waveform is truncated and saved (to reduce file size)
csv_folder = asic.tier[DataTier(:raw_csv), category , period, run]


# convert csv files to lh5 files
csv_to_lh5(asic, period, run, category, channel, csv_folder; csv_heading = 17, nChannels = 1, nwvfmax = NaN, ti = ti)

# read waveforms as sanity check 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
data = read_ldata(asic, DataTier(:raw), filekeys[1], channel)

Table(data)