# Fit Interpretable AI OptimalTreeClassifier (OCT; Bertsimas & Dunn) to predict k-means cluster labels.
# Called from hw6_lib.run_iai_oct:  julia --sysimage=<IAI sys.dylib> -t N iai_oct.jl <workdir>
# Inputs: dims.txt ("n p"), X.bin (n x p Float64 column-major), y.bin (n Float64 cluster ids),
#         configs.txt (one "depth minbucket seed cp" per line). Output: oct_<i>.json (0-based i).
dir = ARGS[1]
n, p = parse.(Int, split(strip(read(joinpath(dir, "dims.txt"), String))))
X = Array{Float64}(undef, n, p); read!(joinpath(dir, "X.bin"), X)
yv = Vector{Float64}(undef, n); read!(joinpath(dir, "y.bin"), yv)
y = Int.(round.(yv))
configs = [split(l) for l in readlines(joinpath(dir, "configs.txt")) if !isempty(strip(l))]
feature_index(f) = parse(Int, replace(string(f), r"^x" => ""))

for (i, c) in enumerate(configs)
    depth, minbucket, seed, cp = parse(Int, c[1]), parse(Int, c[2]), parse(Int, c[3]), parse(Float64, c[4])
    lnr = IAI.OptimalTreeClassifier(max_depth=depth, minbucket=minbucket, cp=cp, random_seed=seed)
    t = @elapsed IAI.fit!(lnr, X, y)
    parts = String[]
    for nd in 1:IAI.get_num_nodes(lnr)
        if IAI.is_leaf(lnr, nd)
            push!(parts, "{\"id\":$nd,\"leaf\":true,\"label\":$(IAI.get_classification_label(lnr, nd))}")
        else
            f = feature_index(IAI.get_split_feature(lnr, nd))
            thr = IAI.get_split_threshold(lnr, nd)
            lo, up = IAI.get_lower_child(lnr, nd), IAI.get_upper_child(lnr, nd)
            push!(parts, "{\"id\":$nd,\"leaf\":false,\"feature\":$f,\"threshold\":$thr,\"lower\":$lo,\"upper\":$up}")
        end
    end
    open(joinpath(dir, "oct_$(i - 1).json"), "w") do io
        write(io, "{\"runtime_s\":$t,\"nodes\":[" * join(parts, ",") * "]}")
    end
    println("config $i depth=$depth minbucket=$minbucket fit=$(round(t, digits=2))s nodes=$(IAI.get_num_nodes(lnr))")
    flush(stdout)
end
