function dt_non_norm(x, mean, sd, df) {
    return 1 / sd * jStat.studentt.pdf( (x - mean) / sd, df);
}

function paired_t_test(x1, x2) {
    var n1 = x1.length;
    var n2 = x2.length;
    mean1 = jStat.mean(x1);
    mean2 = jStat.mean(x2);
    var var1 = Math.pow(jStat.stdev(x1, true), 2);
    var var2 = Math.pow(jStat.stdev(x2, true), 2);
    var sd = Math.sqrt( ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2));
    var t = (mean1 - mean2) / (sd * Math.sqrt(1 / n1 + 1 / n2));
    var p = jStat.ttest(t, n1 + n2 - 2 +1);
    return [mean1 - mean2, t, p];
}


function string_to_num_array(s) {
    s = s.replace(/[^-1234567890.]+$/, '').replace(/^[^-1234567890.]+/, '');
    return jStat.map(s.split(/[^-1234567890.]+/), function(x) {return parseFloat(x)});
}

function histogram_counts(x, breaks, bin_width) {
  var min = jStat.min(x);
  var max = jStat.max(x);
  if(bin_width) {
    var range =  max - min;
    breaks = Math.ceil(range / bin_width)
  }

  var bins = [];
  for (var i = 0; i < breaks; i++) {
    bins.push([min + i / (breaks) * (max - min) + (max - min) / breaks / 2, 0])
  }
  for (var i = 0; i < x.length; i++) {
    bin_i = Math.floor((x[i] - min) / (max - min) * breaks);
    if (isNaN(bin_i)) {
      bin_i = 0
    }
    if (bin_i > breaks - 1) {
      bin_i = breaks - 1
    }
    if (bin_i < 0) {
      bin_i = 0
    }
    bins[bin_i][1]++;
  }
  return bins
}

function HDIofMCMC(x) {
    x = x.slice(0); // Clone x so that we don't mess it up.
    x.sort(function(a,b){return a-b})
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

function sample_from_prior(n, null_prob, rscale) {
  s = [];
  for(var i = 0; i < n; i++) {
    if(Math.random() < null_prob) {
      s.push(0);
    } else {
      s.push(jStat.cauchy.sample(0, rscale));
    }
  }
  return s;
}


function new_sampler(data, null_prob, rscale, lower_rope, upper_rope) {
  var params = {
    mu: {type: "real", init: jStat.mean(data.y1)},
    eff_mu_diff: {type: "real", init: 0},
    is_null: {type: "binary", init: 0},
    log_sigma: {type: "real", init: Math.log(jStat.stdev(data.y1.concat(data.y2))) }};
  
  var log_post = function(s, d) {
    var log_post = 0;
    // Priors
    // implicit unif(-inf, inf) on s.mu
    s.sigma = Math.exp(s.log_sigma)

    log_post += ld.cauchy(s.eff_mu_diff, 0, rscale);
    if(s.is_null) {
      log_post += Math.log(null_prob);
    } else {
      log_post += Math.log(1 - null_prob);
    }
    
    // Likelihood
    var i;
    for(i = 0; i < d.y1.length; i++) {
      log_post += ld.norm(d.y1[i], s.mu, s.sigma);
    }
    s.mu2 = s.mu + !s.is_null * s.sigma * s.eff_mu_diff;      
    for(i = 0; i < d.y2.length; i++) {
      log_post += ld.norm(d.y2[i], s.mu2, s.sigma);
    }
    
    // Adding some derived quantities
    s.is_alt = !s.is_null + 0.0
    s.eff_size = !s.is_null * s.eff_mu_diff;  
    s.mu_diff =  s.mu2 - s.mu
    if(s.eff_size < lower_rope) {
      s.region = -1;  
    } else if(s.eff_size > upper_rope) {
      s.region = 1;
    } else {
      s.region = 0;
    }
    
    return log_post;
  };
  
  return new mcmc.AmwgSampler(params, log_post, data);
}

// Thins the chain and adds indices to every parameter array
function chain_to_plot_data(chain, step_size) {
    var keys = Object.keys(chain);
    var thinned_chain = {};
    for(var key_i = 0; key_i < keys.length; key_i++) {
      var key = keys[key_i];
      var par = chain[key];
      var thin_par = [];
      for(var sample_i = 0; sample_i < par.length; sample_i += step_size) {
          thin_par.push([sample_i, par[sample_i]]);
      }
      thinned_chain[key] = thin_par;
    }
    return thinned_chain;
}

function concat_samples(s1, s2) {
  var keys1 = Object.keys(s1);
  var s3 = {};
  for(var i = 0; i < keys1.length; i++) {
    var key = keys1[i];
    s3[key] = s1[key].concat(s2[key]);
  }
  return s3;
}


function plot_mcmc_chain(div_id, plot_data, title) {
    $.plot($("#" + div_id), [{data: plot_data, label: title}], {shadowSize: 0});
}

function plot_prob_cat(div_id, cat_samples, cats, labels) {
  cat_samples = cat_samples.concat(cats);
  var cat_count = cat_samples.reduce(function(count, val) { count[val] = ++count[val] || 1; return count }, {});
  var cats = Object.keys(cat_count).sort();
  var ticks = [];
  var cat_data = [];
  for(var i = 0; i < cats.length; i++) {
    var cat = cats[i];
    var prob = (cat_count[cat] + 1) / (cat_samples.length + cats.length)
    cat_data.push([parseInt(cat), prob]);
    ticks.push([parseInt(cat), labels[i] +": " + Math.round(prob * 100) + "%"]);
  }
  var plot_options = {font: {size: 9}, shadowSize: 0, xaxis: {ticks: ticks, autoscaleMargin: 0.1 }, yaxis: {min: 0, max: 1.0}};

  $.plot($("#" + div_id), [{data: cat_data, bars: {show: true, align: "center", barWidth: 0.8}}], plot_options);

}


function plot_mcmc_hist(div_id, param_data, show_hdi, comp_value, show_mean_value, rope, xlim, bin_width) {
    var bar_data = histogram_counts(param_data, 29, bin_width);
    var bar_width = bar_data[1][0] - bar_data[0][0];

    var mean = jStat.mean(param_data);
    var mean_data = [[mean, 0]];
    var mean_label = "Mean: " + mean.toPrecision(3);
    var hdi, hdi_data, hdi_label;
    if(show_hdi) {
        hdi = HDIofMCMC(param_data);
        hdi_data = [[hdi[0], 0], [hdi[1], 0]];
        hdi_label = "95% HDI ("+ hdi[0].toPrecision(3) + ", " + hdi[1].toPrecision(3) +")";
    }
    
    var comp_data, comp_perc, comp_label;
    if(comp_value != null) {
        comp_data = [[comp_value, 0], [comp_value, Infinity]];
        comp_perc = perc_larger_and_smaller_than(comp_value, param_data);
        comp_label = "" + (comp_perc[0] * 100).toPrecision(3) + "% < " + comp_value + " < " + (comp_perc[1] * 100).toPrecision(3) + "%";
    }
    var plot_options = {font: {size: 9}, shadowSize: 0, xaxis: {autoscaleMargin: 0.01 }, yaxis: {autoscaleMargin:0.66}};

    if(xlim != null) {
        plot_options.xaxis = {min: xlim[0], max: xlim[1]};
    }

    var plot_spec = [{data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}}];

    if(comp_value != null) {
      plot_spec.push({data: comp_data, label: comp_label, lines: {lineWidth: 2}});
    } else {
      plot_spec.push([]);
    }

    if(show_hdi) {
      plot_spec.push({data: hdi_data, label: hdi_label, lines: {lineWidth: 5}});
    } else {
      plot_spec.push([]);
    }

    if(show_mean_value) {
      plot_spec.push({data: mean_data, label: mean_label, points: { show: true }});
    } else {
      plot_spec.push([]); 
    }

    if(rope != null) {
      var left_rope_data = [[rope[0], 0], [rope[0], Infinity]];
      var right_rope_data = [[rope[1], 0], [rope[1], Infinity]];
      plot_spec.push({data: left_rope_data, color: "#BA55D3", label: "ROPE", lines: {lineWidth: 2}});
      plot_spec.push({data: right_rope_data, color: "#BA55D3",lines: {lineWidth: 2}});
    } else {
      plot_spec.push([]); 
    }

    $.plot($("#" + div_id), plot_spec, plot_options);
}


function plot_mcmc_hist_old(div_id, param_data, show_hdi, comp_value, xlim) {
    var bar_data = histogram_counts(param_data, 30);
    var bar_width = bar_data[1][0] - bar_data[0][0];

    var mean = jStat.mean(param_data);
    var mean_data = [[mean, 0]];
    var mean_label = "Mean: " + mean.toPrecision(3);
    var hdi, hdi_data, hdi_label;
    if(show_hdi) {
        hdi = HDIofMCMC(param_data);
        hdi_data = [[hdi[0], 0], [hdi[1], 0]];
        hdi_label = "95% HDI ("+ hdi[0].toPrecision(3) + ", " + hdi[1].toPrecision(3) +")";
    }
    
    var comp_data, comp_perc, comp_label;
    if(comp_value != null) {
        comp_data = [[comp_value, 0], [comp_value, Infinity]];
        comp_perc = perc_larger_and_smaller_than(comp_value, param_data);
        comp_label = "" + (comp_perc[0] * 100).toPrecision(3) + "% < " + comp_value + " < " + (comp_perc[1] * 100).toPrecision(3) + "%";
    }
    var plot_options = {font: {size: 9}, shadowSize: 0, yaxis: {autoscaleMargin:0.66}};

    if(xlim != null) {
        plot_options.xaxis = {min: xlim[0], max: xlim[1]};
    }
    if(show_hdi && comp_value === null) {
        $.plot($("#" + div_id), [
            {data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}},
            [] ,
            {data: hdi_data, label: hdi_label, lines: {lineWidth: 5}},
            {data: mean_data, label: mean_label, points: { show: true }}], 
          plot_options);
    } else if(! show_hdi && comp_value != null){
        $.plot($("#" + div_id), [
            {data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}},
            {data: comp_data, label: comp_label, lines: {lineWidth: 2}},
            {data: mean_data, label: mean_label, points: { show: true }}],
          plot_options);
    } else if(show_hdi && comp_value != null){
        $.plot($("#" + div_id), [
            {data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}},
            {data: comp_data, label: comp_label, lines: {lineWidth: 2}},
            {data: hdi_data, label: hdi_label, lines: {lineWidth: 5}},
            {data: mean_data, label: mean_label, points: { show: true }}],
          plot_options);
    }else {
        $.plot($("#" + div_id), [
            {data: bar_data, bars: {show: true, align: "center", barWidth: bar_width}},
            {data: mean_data, label: mean_label, points: { show: true }}],
          plot_options);
    }
}