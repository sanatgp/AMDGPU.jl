@testitem "core" setup=[TSCore] begin

using AMDGPU: HIP, Runtime, Device, Mem
import AMDGPU: @allowscalar

AMDGPU.allowscalar(false)

macro grab_output(ex, io=stdout)
    quote
        mktemp() do fname, fout
            ret = nothing
            open(fname, "w") do fout
                if $io == stdout
                    redirect_stdout(fout) do
                        ret = $(esc(ex))
                    end
                elseif $io == stderr
                    redirect_stderr(fout) do
                        ret = $(esc(ex))
                    end
                end
            end
            ret, read(fname, String)
        end
    end
end

@testset "HIPDevice" begin
    @testset "Device props" begin
        devices = AMDGPU.devices()
        for (idx, device) in enumerate(devices)
            @test AMDGPU.device_id(device) == idx

            device_name = HIP.name(device)
            @test length(device_name) > 0

            @test occursin("gfx", HIP.gcn_arch(device))
            @test HIP.wavefront_size(device) in (32, 64)
        end
    end
end

include("codegen/synchronization.jl")
include("codegen/trap.jl")

include("rocarray/base.jl")
include("rocarray/broadcast.jl")

include("tls.jl")

end