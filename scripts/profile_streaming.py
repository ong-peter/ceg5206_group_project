"""Instrument temporary copies to estimate score and online-update stage costs.

These intrusive diagnostics are separate from uninstrumented benchmark timings.
Only serial execution is supported. No production source file is modified.
"""
import csv
from pathlib import Path
import shlex
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
from check import inputs, reference
import numpy as np


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("Profiling insertion point changed; update the diagnostic script")
    return source.replace(old, new, 1)


def instrument(source):
    source = "#include <chrono>\n" + source
    source = replace_once(source, "namespace {", """
double profile_scores_ms = 0, profile_update_ms = 0;
std::size_t profile_tiles = 0, profile_maximum_increases = 0;
namespace {
using ProfileClock = std::chrono::steady_clock;
""")
    point = "        const std::size_t columns = std::min(work.bc, n - key);"
    source = replace_once(source, point, point + "\n        const auto score_start = ProfileClock::now();")
    point = "        for (std::size_t i = 0; i < rows; ++i) {\n            float* row = scores + i * work.bc;"
    source = replace_once(source, point, """
        profile_scores_ms += std::chrono::duration<double, std::milli>(ProfileClock::now() - score_start).count();
        const auto update_start = ProfileClock::now();
""" + point)
    point = "            const float next_maximum = std::max(maximum[i], block_maximum);"
    source = replace_once(source, point, point + """
            if (key != 0) {
                ++profile_tiles;
                if (next_maximum > maximum[i]) ++profile_maximum_increases;
            }
""")
    point = "        key += columns;"
    source = replace_once(source, point, """
        profile_update_ms += std::chrono::duration<double, std::milli>(ProfileClock::now() - update_start).count();
""" + point)
    return source


def main():
    output = ROOT / "results/optimization"
    output.mkdir(parents=True, exist_ok=True)
    driver = r'''
#include "attention.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
extern double profile_scores_ms, profile_update_ms;
extern std::size_t profile_tiles, profile_maximum_increases;
template<class T> std::vector<T> read(const char* path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(count*sizeof(T)))
        throw std::runtime_error("Incorrect profile input size");
    file.seekg(0);
    std::vector<T> result(count);
    file.read(reinterpret_cast<char*>(result.data()), count*sizeof(T));
    if (!file) throw std::runtime_error("Could not read profile input");
    return result;
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::invalid_argument("profile N d inputs reference");
        const auto n=std::stoull(argv[1]), d=std::stoull(argv[2]);
        auto data=read<float>(argv[3],3*n*d);
        auto expected=read<double>(argv[4],n*d);
        std::vector<float> out(n*d), scratch(attention_streaming_scratch(n,d,{}));
        auto call=[&] {attention_streaming(data.data(),data.data()+n*d,data.data()+2*n*d,out.data(),scratch.data(),n,d,{});};
        for (int i=0;i<5;++i) call();
        profile_scores_ms=profile_update_ms=0;
        profile_tiles=profile_maximum_increases=0;
        auto start=Clock::now();
        for (int i=0;i<5;++i) call();
        auto total=std::chrono::duration<double,std::milli>(Clock::now()-start).count()/5;
        for (std::size_t i=0;i<n*d;++i)
            if (!std::isfinite(out[i]) || std::abs(out[i]-expected[i])>1e-4+1e-3*std::abs(expected[i]))
                throw std::runtime_error("Profile correctness failed");
        std::cout<<std::setprecision(10)<<profile_scores_ms/5<<','<<profile_update_ms/5<<','<<total<<','
                 <<profile_tiles/5<<','<<profile_maximum_increases/5<<'\n';
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
'''
    compiler=shlex.split(os.environ.get("CXX", "c++"))
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        temp=Path(tmp)
        (temp/"profile.cpp").write_text(driver)
        for name,path in [("original","src/benchmarks/streaming_original.cpp"),("optimized","src/streaming.cpp")]:
            (temp/"instrumented.cpp").write_text(instrument((ROOT/path).read_text()))
            binary=temp/f"profile-{name}"
            subprocess.run([*compiler,"-O3","-std=c++17","-I",str(ROOT/"src"),str(temp/"profile.cpp"),str(temp/"instrumented.cpp"),"-o",str(binary)],check=True)
            for n,d in [(512,64),(2048,64)]:
                q,k,v=inputs(n,d,n+d)
                np.concatenate([q.ravel(),k.ravel(),v.ravel()]).tofile(temp/"inputs.bin")
                reference(q,k,v).tofile(temp/"reference.bin")
                result=subprocess.check_output([str(binary),str(n),str(d),str(temp/"inputs.bin"),str(temp/"reference.bin")],text=True)
                scores,update,total,tiles,increases=result.strip().split(",")
                rows.append(dict(algorithm=name,n=n,d=d,score_and_pack_ms=scores,online_update_and_pv_ms=update,
                                 instrumented_total_ms=total,later_row_tiles=tiles,maximum_increases=increases))
                print(rows[-1],flush=True)
    with (output/"profile.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


if __name__ == "__main__":
    main()
