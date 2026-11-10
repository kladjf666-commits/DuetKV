"""Build from the A800-only source fork and the two previously shipped test archives."""
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

A800 = "bc5b8e017144f6e7efea92cb5e224e0f32fc0c1ffd6f03ce1143e3433143f248"
RTX = "aee47250e06e790f0589c6841d3ad680c4238827cf669353a0cf031aeabe574c"


def collect(project, scripts):
    fork = scripts.parent / "PhaseKV"
    output = project / "PhaseKV与Full消融"
    assets = {p.name: p.read_bytes() for p in (scripts/"launch.py",scripts/"probe.py")}
    for p in fork.rglob("*.py"):
        assets["core/PhaseKV/"+p.relative_to(fork).as_posix()] = p.read_bytes()
    if len(assets) != 18:
        raise ValueError("Expected 16 core files in the A800 fork")
    for fname, expected, entries in [
        ("PhaseKV_A800_80G_paper_v1.zip", A800,
         {n:"phasekv_a800_s1/"+n for n in ("engine.py","batch_backend.py","data/speed_rows.jsonl.gz",
          "native/config.json","native/native_runtime.py","native/ablation_runtime.py","native/ablation_attention.py",
          "native/ablation_kernels.py","native/runtime/phasekv_llama_attention.py")}),
        ("PhaseKV_4090_batch2_6.zip", RTX,
         {f"legacy/{n}":"phasekv_4090_batch2_6/"+n for n in ("batch_speed.py","base/quick.py")})]:
        archive=output/fname
        if not archive.is_file():
            archive=project/"PhaseKVA800测试"/"与其它算法的对比"/fname
        if sha256(archive.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Unexpected dependency archive: {archive}")
        with ZipFile(archive) as z:
            assets.update({name:z.read(member) for name,member in entries.items()})
    for n,data in assets.items():
        if n.endswith(".py"):
            compile(data,n,"exec")
    assets["SHA256SUMS"]="".join(f"{sha256(data).hexdigest()}  {n}\n" for n,data in sorted(assets.items())).encode()
    return assets


def build(project, scripts, target):
    assets=collect(project,scripts)
    prefix="a800_prompt_mma_v1/" if target.name == "PhaseKV_A800_prompt_mma_v1.zip" else "a800_reason_v1/"
    tmp=target.with_suffix(".partial")
    with ZipFile(tmp,"w",compression=ZIP_DEFLATED) as z:
        for name,data in assets.items():
            z.writestr(prefix+name,data)
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        assert all(z.read(prefix+name)==data for name,data in assets.items())
    tmp.replace(target)
    print(target,target.stat().st_size,sha256(target.read_bytes()).hexdigest())


if __name__=="__main__":
    scripts=Path(__file__).resolve().parent
    project=scripts.parent.parent
    output=project/"PhaseKVA800测试"/"与Full消融"
    if not output.is_dir():
        output=project/"PhaseKV与Full消融"
    build(project,scripts,output/"PhaseKV_A800_prompt_mma_v1.zip")
