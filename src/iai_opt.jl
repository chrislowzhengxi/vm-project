# Fit Interpretable AI OptimalTreePolicyMinimizer (Optimal Policy Trees, Amram, Dunn & Zhuo 2022) for each config.
# Called from final_lib.run_iai_policy_trees:  julia --sysimage=<IAI sys.dylib> -t N iai_opt.jl <workdir>
# Inputs in <workdir>: dims.txt ("n p"), X.bin (n x p Float64, column-major), C.bin (n x 4 Float64 costs),
#                      configs.txt (one "depth minbucket seed" per line).
# Output: tree_<i>.json per config (0-based i), with nodes (1-based ids) and the fit runtime.
dir = ARGS[1]
n, p = parse.(Int, split(strip(read(joinpath(dir, "dims.txt"), String))))
X = Array{Float64}(undef, n, p); read!(joinpath(dir, "X.bin"), X)
C = Array{Float64}(undef, n, 4); read!(joinpath(dir, "C.bin"), C)
configs = [parse.(Int, split(l)) for l in readlines(joinpath(dir, "configs.txt")) if !isempty(strip(l))]

treatment_index(t) = t isa Integer ? Int(t) : parse(Int, replace(string(t), r"^x" => ""))  # cost columns x1..x4
feature_index(f) = parse(Int, replace(string(f), r"^x" => ""))

for (i, (depth, minbucket, seed)) in enumerate(configs)
    lnr = IAI.OptimalTreePolicyMinimizer(max_depth=depth, minbucket=minbucket, cp=1e-6, random_seed=seed)
    t = @elapsed IAI.fit!(lnr, X, C)
    parts = String[]
    for nd in 1:IAI.get_num_nodes(lnr)
        if IAI.is_leaf(lnr, nd)
            a = treatment_index(IAI.get_policy_treatment_rank(lnr, nd)[1])
            push!(parts, "{\"id\":$nd,\"leaf\":true,\"action\":$a}")
        else
            f = feature_index(IAI.get_split_feature(lnr, nd))
            thr = IAI.get_split_threshold(lnr, nd)
            lo, up = IAI.get_lower_child(lnr, nd), IAI.get_upper_child(lnr, nd)
            push!(parts, "{\"id\":$nd,\"leaf\":false,\"feature\":$f,\"threshold\":$thr,\"lower\":$lo,\"upper\":$up}")
        end
    end
    open(joinpath(dir, "tree_$(i - 1).json"), "w") do io
        write(io, "{\"runtime_s\":$t,\"version\":\"IAI system image v3.2.2\",\"nodes\":[" * join(parts, ",") * "]}")
    end
    println("config $i depth=$depth minbucket=$minbucket seed=$seed fit=$(round(t, digits=2))s nodes=$(IAI.get_num_nodes(lnr))")
    flush(stdout)
end
