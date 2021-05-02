using BenchmarkTools, AutoARIMA

const SUITE = BenchmarkGroup()

z = cos.(1:1000000) .* log.(1:1000000) .* randn()

SUITE["fit"] = BenchmarkGroup()

SUITE["fit"]["ar"] = BenchmarkGroup()
for f in (yule_walker, levinson_durbin, least_squares)
    for p in (1, 10)
        SUITE["fit"]["ar"][string(f),p] = @benchmarkable $f($z, $p)
    end
end

SUITE["fit"]["ma"] = BenchmarkGroup()
for f in (innovations,)
    for q in (1, 10)
        SUITE["fit"]["ma"][string(f),q] = @benchmarkable $f($z, $q, m=100)
    end
end

SUITE["fit"]["arma"] = BenchmarkGroup()
for f in (hannan_rissanen,)
    for p in (1, 10), q in (1, 10)
        SUITE["fit"]["arma"][string(f),p,q] = @benchmarkable $f($z, $p, $q)
    end
end

SUITE["forecast"] = BenchmarkGroup()

SUITE["forecast"]["ar"] = BenchmarkGroup()
for p in (1, 10, 100)
    ar = fit(ARParams(p), z)
    SUITE["forecast"]["ar"][p] = @benchmarkable forecast($ar, $z)
end

SUITE["forecast"]["ma"] = BenchmarkGroup()
for q in (1, 10, 100)
    ma = fit(MAParams(q), z, n=200)
    SUITE["forecast"]["ma"][q] = @benchmarkable forecast($ma, $z)
end

SUITE["forecast"]["arma"] = BenchmarkGroup()
for p in (1, 10, 100), q in (1, 10, 100)
    arma = fit(ARMAParams(p, q), z)
    SUITE["forecast"]["arma"][p,q] = @benchmarkable forecast($arma, $z)
end
