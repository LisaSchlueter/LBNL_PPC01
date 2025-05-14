
using LegendDataManagement
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using LegendDSP
using LegendPlots
using CairoMakie
using Unitful, Dates
using Measurements: value as mvalue, uncertainty as muncert 
using TypedTables
using StatsBase

relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__)
include("$relPath/hpge-ana/utils/utils_aux.jl")
include("$relPath/hpge-ana/utils/utils_physics.jl")
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(2)
channel = ChannelId(1)
category = DataCategory(:cal)


# Skutek Digitizer "VIREO FemtoDAQ"
resolution_ADC = 2^14
dynamicrange_V = 2.0
V_per_ADC = dynamicrange_V/resolution_ADC


# load calibration curve
cal_func_str = asic.par.rpars.ecal[period, run, channel].e_trap.cal.func
cal_func = ljl_propfunc(cal_func_str)

e_ADC = 0:1:3000
e_cal = cal_func.(Table(e_trap = e_ADC))

fig = Figure(size = (500, 600))
ax = Axis(fig[1, 1],  xlabel = "Pulse height (ADC)", ylabel = "Pulse height (keV)", xgridvisible = true,  ygridvisible = true, limits = (0, maximum(e_ADC), 0, nothing))
lines!(ax,  e_ADC, ustrip.(e_cal),
    label = "Calibration curve: 1000 ADC ≈ $(round(Int, ustrip(asic.par.rpars.ecal[period, run, channel].e_trap.cal.par[1] + 1000*mvalue(asic.par.rpars.ecal[period, run, channel].e_trap.cal.par[2])))) keV")
axislegend(ax, position = :lt)
ax2 = Axis(fig[2, 1],  xlabel = "Pulse height (V)", ylabel = "Pulse height (keV)",  xgridvisible = true,  ygridvisible = true,  yminorgridvisible = true, limits = (0, maximum(e_ADC*V_per_ADC), 0, nothing))
lines!(ax2, e_ADC*V_per_ADC, ustrip.(e_cal), label = "14-bit resolution:  1000 ADC ≈ $(round(1000 * V_per_ADC, digits = 2)) V")
axislegend(ax2, position = :lt)
fig
plt_name = "$(@__DIR__)/plots/CalibrationCurve_VireoFemtoDAQ_14bit_$run.png"
save(plt_name, fig)


# calculate detector capacitance: 
# Q = C * V
# estimate charge fron reconstructed energy deposition 
gain = 2 # gain of CSA
e_ADC = 0:500:3000 # energy in ADC 
Vout_V = e_ADC .* V_per_ADC # convert ADC to V using dynamic range of the system
Vsig_V = Vout_V / gain # voltage pulse height at the input of the CSA / output of detector
e_cal = cal_func.(Table(e_trap = e_ADC)) # energy in keV --> calibration function from data analysis/gamma spectrum
Temp = 90
NumChargeCarrier = Ge_NumberChargeCarrier.(ustrip.(uconvert.(u"eV", e_cal)), Temp)
Charge_C = NumChargeCarrier * electron_charge

Q_fC = 1e15 * Charge_C

Table(
    E_ADC = e_ADC, 
    E_keV = ustrip.(e_cal),
    Vs_V = Vsig_V,
    Q_fC = 1e15 * Charge_C,
    Cf_pF = 1e12 .* (Charge_C ./ Vsig_V), 
)
