# Rotor37 Turbomachinery

Repository to implement a point-cloud based GNN regression pipeline to predict per-vertx flow quantities.

- Python 3.8+, CUDA 11.7+
- PyTorch, PyGeometric, NumPy
- Meshio
- Paraview

How to Run:

python compute_stats.py
python newmain.py

/vols/numeca_nfs01/devenv/tools/cfd_devops/farm_scripts/run_on_farm.sh -farm /vols/numeca_nfs01/dling/projects/runrotor.lsf -farm-nbgpu=1 -farm-submit-opts="-m sj-numecagpu-01 -gpu num=1" -farm-notail&


