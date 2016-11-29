function dt_non_norm(x, mean, sd, df) {
    return 1 / sd * jStat.studentt.pdf( (x - mean) / sd, df)
}

function paired_t_test(x1, x2) {
    var n1 = x1.length
    var n2 = x2.length
    mean1 = jStat.mean(x1)
    mean2 = jStat.mean(x2)
    var var1 = Math.pow(jStat.stdev(x1, true), 2)
    var var2 = Math.pow(jStat.stdev(x2, true), 2)
    var sd = Math.sqrt( ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    var t = (mean1 - mean2) / (sd * Math.sqrt(1 / n1 + 1 / n2))
    var p = jStat.ttest(t, n1 + n2 - 2 +1)
    return [mean1 - mean2, t, p]
}


function string_to_num_array(s) {
    s = s.replace(/[^-1234567890.]+$/, '').replace(/^[^-1234567890.]+/, '')
    return jStat.map(s.split(/[^-1234567890.]+/), function(x) {return parseFloat(x)})
}

function histogram_counts(x, breaks) {
    var min = jStat.min(x)
    var max = jStat.max(x)
    var bins = []
    for(var i =0; i < breaks; i++) {bins.push([min + i/(breaks) * (max - min) + (max - min) / breaks / 2, 0])}
    for(var i = 0; i < x.length; i++) {
        bin_i = Math.floor((x[i] - min) / (max - min) * breaks)
        if(isNaN(bin_i)) {bin_i = 0}
        if(bin_i > breaks - 1) {bin_i = breaks - 1}
        if(bin_i < 0) {bin_i = 0}
        bins[bin_i][1]++
    }
    return bins
}

function HDIofMCMC(x) {
    x = x.sort(function(a,b){return a-b})
    var ci_nbr_of_points = Math.floor(x.length * 0.95)
    var min_width_ci = [jStat.min(x), jStat.max(x)] // just initializing
    for(var i = 0; i < x.length - ci_nbr_of_points; i++) {
        var ci_width = x[i + ci_nbr_of_points] - x[i]
        if(ci_width < min_width_ci[1] - min_width_ci[0]) {
            min_width_ci = [x[i], x[i + ci_nbr_of_points]]
        }
    }
    return min_width_ci
}

function perc_larger_and_smaller_than(comp, data) {
    comps = jStat.map( data, function( x ) {
        if(x >= comp) {
            return 1
        } else {
            return 0
        }
    })
    mean_larger = jStat.mean(comps)
    return [1 - mean_larger, mean_larger]
}

function chain_to_plot_data(chain, step_size, samples_to_keep) {
    if(samples_to_keep != null) {
        step_size = chain.length / samples_to_keep
    } 
    plot_data = []
    for(var i = 0; i < chain[0].length; i++) {
        plot_data.push([])
    }
    for(var i = 0; i < chain.length; i += step_size) {
        var sample_i = Math.floor(i)
        var sample = chain[sample_i]
        for(var param_i = 0; param_i < sample.length; param_i++) {
            plot_data[param_i].push([sample_i, sample[param_i]])
        }
    }
    return plot_data
}

function param_chain(chain, param_i) {
    var param_data = []
    for (var i = 0; i < chain.length; i++) {
        param_data.push(chain[i][param_i])
    }
    return param_data
}

// Constructor for the adaptive metropolis within Gibbs
function amwg(start_values, posterior, data_calc) {
    var n_params = start_values.length
    var batch_count = 0
    var batch_size = 50
    var chain = []
    var curr_state = start_values
    var log_sd = []
    this.log_sd = log_sd
    var acceptance_count = []
    var running_asynch = false
    for (var i = 0; i < n_params; i++) {
        log_sd[i] = 0
        acceptance_count[i] = 0 
    }

    function next_sample() {
        if(data_calc != null) {
            chain.push(curr_state.concat(data_calc(curr_state)))
        } else {
            chain.push(curr_state)
        }

        for(var param_i = 0; param_i < n_params; param_i++) {
            var param_prop = jStat.normal.sample(curr_state[param_i] , Math.exp( log_sd[param_i]) )
            var prop = curr_state.slice()
            prop[param_i] = param_prop
            try {
                var curr_post_dens = posterior(curr_state)
                var prop_post_dens = posterior(prop)
                if(! isFinite(curr_post_dens)) {
                    // if curr_post_dens is as bad as, say, negative infinity or NaN we should always jump
                    var accept_prob = 1
                } else { 
                    var accept_prob = Math.exp(prop_post_dens - curr_post_dens)
                }
            } catch(err) { // Probably SD < 0 or similar...
                var accept_prob = 0
            }
            if(accept_prob > Math.random()) {
                acceptance_count[param_i]++
                curr_state = prop
            } // else do nothing
        }

        if(chain.length % batch_size == 0) {
            batch_count++
            for(var param_i = 0; param_i < n_params; param_i++) {
                if(acceptance_count[param_i] / batch_size > 0.44) {
                    log_sd[param_i] += Math.min(0.01, 1/Math.sqrt(batch_count))
                } else if(acceptance_count[param_i] / batch_size < 0.44) {
                    log_sd[param_i] -= Math.min(0.01, 1/Math.sqrt(batch_count))
                }
                acceptance_count[param_i] = 0 
            }
        }
        return curr_state
    }

    this.next_sample = next_sample

    this.get_chain = function() {return chain}
    this.get_curr_state = function() {return curr_state}
    this.get_curr_post_dens = function() {return posterior(curr_state)}

    this.burn = function(n) {
        var temp_chain = chain.slice()
        this.n_samples(n)
        chain = temp_chain
    }

    function n_samples(n) {
        for(var i = 0; i < n - 1; i++) {
            next_sample()
        }
        return next_sample()
    }

    this.n_samples = n_samples

    this.is_running_asynch = function() {return running_asynch}

    function n_samples_asynch(n, nbr_of_samples) {
        if(n > 0) {
            running_asynch = true
            n_samples(nbr_of_samples)
            return setTimeout(function() {n_samples_asynch(n - nbr_of_samples, nbr_of_samples)}, 0)
        } else {
            running_asynch = false
        }
    }

    this.n_samples_asynch = n_samples_asynch
}

function make_BEST_posterior_func(y1, y2) {
    data = [y1, y2]
    mean_mu = jStat.mean(y1.concat(y2))
    sd_mu = jStat.stdev(y1.concat(y2)) * 1000000
    sigma_low = jStat.stdev(y1.concat(y2)) / 1000
    sigma_high = jStat.stdev(y1.concat(y2)) * 1000


    var posterior = function(params) {
        var mu = [params[0], params[1]]
        var sigma = [params[2], params[3]]
        var nu = params[4]
        var log_p = 0
        log_p += Math.log(jStat.exponential.pdf( nu - 1, 1/29 ))
        for(var group = 0; group < 2; group++) {
            log_p += Math.log(jStat.uniform.pdf( sigma[group], sigma_low, sigma_high ))
            log_p += Math.log(jStat.normal.pdf( mu[group], mean_mu, sd_mu ))
            for(var subj_i = 0; subj_i < data[group].length; subj_i++) {
                log_p += Math.log(dt_non_norm(data[group][subj_i], mu[group], sigma[group], nu ))
            }
        }
        return log_p
    }

    return posterior
}

function plot_mcmc_chain(div_id, plot_data, title) {
    $.plot($("#" + div_id), [{data: plot_data, label: title}], {shadowSize: 0})
}

function plot_mcmc_hist(div_id, param_data, show_hdi, comp_value, xlim) {
    var bar_data = histogram_counts(param_data, 30)
    var bar_width = bar_data[1][0] - bar_data[0][0]

    var mean = jStat.mean(param_data)
    var mean_data = [[mean, 0]]
    var mean_label = "Mean: " + mean.toPrecision(3)
    if(show_hdi) {
        var hdi = HDIofMCMC(param_data)
        var hdi_data = [[hdi[0], 0], [hdi[1], 0]]
        var hdi_label = "95% HDI ("+ hdi[0].toPrecision(3) + ", " + hdi[1].toPrecision(3) +")"
    }

    if(comp_value != null) {
        var comp_data = [[comp_value, 0], [comp_value, Infinity]]
        var comp_perc = perc_larger_and_smaller_than(comp_value, param_data)
        var comp_label = "" + (comp_perc[0] * 100).toPrecision(3) + "% < " + comp_value + " < " + (comp_perc[1] * 100).toPrecision(3) + "%"

    }
    var plot_options = {font: {size: 9}, shadowSize: 0, yaxis: {autoscaleMargin:0.66}}

    if(xlim != null) {
        plot_options["xaxis"] = {min: xlim[0], max: xlim[1]}
    }
    if(show_hdi && comp_value == null) {
        $.plot($("#" + div_id), [{data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}},[] , {data: hdi_data, label: hdi_label, lines: {lineWidth: 5}}, {data: mean_data, label: mean_label, points: { show: true }}], plot_options)
    } else if(! show_hdi && comp_value != null){
        $.plot($("#" + div_id), [{data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}}, {data: comp_data, label: comp_label, lines: {lineWidth: 2}}, {data: mean_data, label: mean_label, points: { show: true }}], plot_options)
    } else if(show_hdi && comp_value != null){
        $.plot($("#" + div_id), [{data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}}, {data: comp_data, label: comp_label, lines: {lineWidth: 2}}, {data: hdi_data, label: hdi_label, lines: {lineWidth: 5}}, {data: mean_data, label: mean_label, points: { show: true }}], plot_options)
    }else {
        $.plot($("#" + div_id), [{data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}}, {data: mean_data, label: mean_label, points: { show: true }}], plot_options)
    }
}