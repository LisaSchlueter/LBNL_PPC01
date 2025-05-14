using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using StatsBase
using LegendSpecFits
using Unitful
using Makie, LegendMakie, CairoMakie   
using Distributions
using Measurements
using Measurements: value as mvalue, uncertainty as muncert

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/utils/utils_physics.jl")
include("$(@__DIR__)/$relPath/src/fit_linearity.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(1)
channel = ChannelId(1)
category = DataCategory(:bch)
det = _channel2detector(asic, channel)
# runs = vcat(1:4, 7:10)
runs = 23:32

# load peakfit results 
µ_fit      = [asic.par[category].rpars.ecal[period, DataRun(run), channel].µ for run in runs]
σ_fit   = [asic.par[category].rpars.ecal[period, DataRun(run), channel].fit.σ for run in runs] 
pulser_fit = [asic.par[category].rpars.ecal[period, DataRun(run), channel].µ_pulser for run in runs]

# linear fit 
result, report_lin  = fit_linearity(1, pulser_fit, µ_fit)


begin
    # input for plot:
    report = report_lin 
    color_fit = LegendMakie.AchatBlue
    plot_ribbon = true
    xerrscaling = 1 
    yerrscaling = 1 
    xtickformat =  Makie.automatic
    xticks = Makie.WilkinsonTicks(6, k_min = 5)
    color_res = :black
    markersize = 8
    filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, DataRun(runs[1])])[1]
    show_residuals = true
    show_gof = false 
    units = :MeV
    res_mode = :percent
    xlims = (nothing, nothing)
    ylims_up = (nothing, nothing)
   
    if res_mode == :abs
        if units == :MeV
            ylims_res = (-7, 7)
            yticks_res = [-3, 0 , 3]
        else
            ylims_res = (-8, 8)
            yticks_res = [-4, 0 , 4]
        end
      
    elseif res_mode == :percent
            ylims_res = (-0.4, 0.4)
            yticks_res = -0.2:0.2:0.2
        end 

    xlabel = "Pulser ($units)"
    ylabel = "Signal ($units)"

    bench_set = (
        Cinj_F = 500 * 1e-15, # pulser injection cap 
        gain = 1.0, # csa gain
        bits = 14, 
        dynamicrange_V = 2.0
    )

    data_label_ext = if Timestamp(split(string(filekey), "-")[end]).unixtime < 1741421445
        " (295 K)"
    else
        " (90 K)" 
    end


    # gather values that will be plotted
    x = report.x
    y = report.y
    y_fit = report.f_fit.(x)
    if units == :MeV
        x = pulser_ADC_to_keV.(mvalue.(x), bench_set.Cinj_F; bits = bench_set.bits,  dynamicrange_V = bench_set.dynamicrange_V, gain = bench_set.gain) /1e3
        y = pulser_ADC_to_keV.(mvalue.(y), bench_set.Cinj_F; bits = bench_set.bits,  dynamicrange_V = bench_set.dynamicrange_V, gain = bench_set.gain) /1e3
        y_fit = pulser_ADC_to_keV.(mvalue.(y_fit), bench_set.Cinj_F; bits = bench_set.bits,  dynamicrange_V = bench_set.dynamicrange_V, gain = bench_set.gain) /1e3
    end 

    title = get_plottitle(filekey, det, "Linearity")
    title = replace(title, string(DataRun(runs[1])) => "$(DataRun(runs[1])) to $(DataRun(runs[end]))")
    fit_label = "Linear fit" # fit_label = "Best Fit" * ((!isempty(report.gof) && show_gof) ? " (p = $(round(report.gof.pvalue, digits=2))| χ²/ndf = $(round(report.gof.chi2min, digits=2)) / $(report.gof.dof))" : "")
    data_label = "Data" * data_label_ext
    fig = Figure(size = (600, 750) )
    g = Makie.GridLayout(fig[2,1])
    ax = Makie.Axis(g[1,1], limits = (xlims, ylims_up);
        title =  title, titlesize = 16, xlabel =  xlabel, ylabel = ylabel, xticks = xticks, xtickformat = xtickformat)

    # plot fit
    Makie.lines!(ax, mvalue.(x), mvalue.(y_fit), label = fit_label, color = color_fit)
    if plot_ribbon
        Δyfit = muncert.(y_fit)
        Makie.band!(ax, mvalue.(x), mvalue.(y_fit) .- Δyfit, mvalue.(y_fit) .+ Δyfit, color = (color_fit, 0.2))
    end

    # scatter points with error bars
    Makie.errorbars!(ax, mvalue.(x), mvalue.(y), muncert.(x) .* xerrscaling, direction = :x, color = :black)
    Makie.errorbars!(ax, mvalue.(x), mvalue.(y), muncert.(y) .* yerrscaling, color = :black)
    Makie.scatter!(ax, mvalue.(x), mvalue.(y), marker = :circle, color = :black, label = data_label)
    axislegend(ax, position = :lt)

    if !isempty(report.gof) && show_residuals
        ax.xticklabelsize = 0
        ax.xticksize = 0
        ax.xlabel = ""

        if res_mode == :norm 
            res = report.gof.residuals_norm
            ylabel_res = "Residuals (σ)"
        elseif res_mode == :abs
            res, ylabel_res = if units ==:MeV
                1e3.*mvalue.(y .- y_fit), "Residuals (keV)"
            else
                mvalue.(y .- y_fit), "Residuals ($units)"
            end 
        elseif res_mode == :percent 
            res = 100 .* mvalue.(y .- y_fit)./ (0.5 .*  mvalue.(y .+ y_fit))
            ylabel_res = "Residuals (%)"
        else
            error("Unknown residual mode: $res_mode")
        end

            ax2 = Makie.Axis(g[2,1], yticks = yticks_res, limits = (xlims,ylims_res); xlabel, xticks, xtickformat, ylabel = ylabel_res)
            Makie.hspan!(ax2, [minimum(yticks_res)], [maximum(yticks_res)], color = :silver, alpha = 0.7)
            Makie.hlines!(ax2, [0], [1], color = :silver, linestyle = :solid, linewidth = 2)
            Makie.scatter!(ax2, mvalue.(x), res, color = color_res, markersize = markersize)

            
            # link axis and put plots together
            Makie.linkxaxes!(ax, ax2)
            Makie.rowgap!(g, 0)
            Makie.rowsize!(g, 1, Makie.Auto(4))

            # align ylabels
            yspace = maximum(Makie.tight_yticklabel_spacing!, (ax, ax2))
            ax.yticklabelspace = yspace
            ax2.yticklabelspace = yspace
    end 
    fig

    plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :linearity) * "/"
    pname = plt_folder * _get_pltfilename(asic, filekey, channel, Symbol("linearity_$(res_mode)_$units"))
    save(pname, fig)
    @info "Save peak fit plot to $pname"
    fig
end 


