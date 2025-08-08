# compute_stats.py

import os, glob
import numpy as np

def main():
    root       = "/vols/numeca_nfs01/dling/rotor"
    csv_folder = os.path.join(root, 'csv')

    # FLOW-FIELD STATS 
    all_fields = []
    for fn in glob.glob(os.path.join(csv_folder, 'Blade_*.csv')):
        arr = np.loadtxt(fn, delimiter=',', skiprows=1)[:, :4]
        all_fields.append(arr)
    all_fields = np.vstack(all_fields)         # (total_points, 4)
    field_mean = all_fields.mean(axis=0)       # (4,)
    field_std  = all_fields.std(axis=0)        # (4,)

    # BOUNDARY-CONDITION STATS 
    bc_names = [
        "Static_pressure",
        "Static_pressure_ratio",
        "Static_temperature_ratio",
        "Isentropic_efficiency",
        "Polytropic_efficiency",
        "Absolute_total_pressure_ratio",
        "Absolute_total_temperature_ratio"
    ]
    #load each .npy into a dict
    bc_dicts = {
        name: np.load(os.path.join(root,'Scalars_input', 'Dict', f"{name}.npy"),
                      allow_pickle=True).item()
        for name in bc_names
    }

    blade_ids = sorted(bc_dicts[bc_names[0]].keys())
    bc_rows   = []
    for bid in blade_ids:

        row = [ float(bc_dicts[name][bid]) for name in bc_names ]
        bc_rows.append(row)

    bc_mat  = np.array(bc_rows, dtype=np.float64)  # (n_blades, 7)
    bc_mean = bc_mat.mean(axis=0)                  # (7,)
    bc_std  = bc_mat.std(axis=0)                   # (7,)

    out_path = os.path.join(root, 'field_bc_stats.npz')
    np.savez(out_path,
             field_mean=field_mean, field_std=field_std,
             bc_mean   =bc_mean,    bc_std   =bc_std,
             blade_ids =blade_ids)

    # Print summary
    print(f"Wrote “{out_path}” with:")
    for name, m, s in zip(['Density','Entropy','Pressure','Temp'],
                          field_mean, field_std):
        print(f"  {name:10s} mean={m:.3f}, std={s:.3f}")

    print("\nBoundary-condition stats:")
    for name, m, s in zip(bc_names, bc_mean, bc_std):
        print(f"  {name:30s} mean={m:.3f}, std={s:.3f}")

if __name__ == "__main__":
    main()
