"""Read-only diagnostic: why no 90-deg stop exists below the truck-front pallet (photos 4, 6 A; 7 B). Writes nothing."""
import sys
import numpy as np
sys.path.insert(0, "/home/binci/vlm_ws/semantic_task_navigation/experiments/navigation_handoff/scripts")
import path_planning_batch_step4b as B
CT, ME, OP, WB = B.CT, B.ME, B.OP, B.WB
cfg = CT.load_config()
frame, fc, run = OP.load_step2_frame(cfg)
g = fc["grid"]; grid = CT.MapGrid(frame, g["extent_floor_cm"], g["resolution_m"])
pgm, _ = ME.read_pgm(run / "maps/map.pgm"); layers = np.load(run / "maps/map_layers.npy")
urdf = WB.read_urdf_wheels(cfg["wheel_boundary_experiment"]["urdf"]); nominal = (cfg["robot"]["length_cm"], cfg["robot"]["width_cm"])
inv = cfg["step4a"]["inventory"]; px = cfg["planning"]["raster_px_per_cm"]
for photo, box in (("photo_4", inv["photo_4"]["cube"]), ("photo_6", inv["photo_6"]["cube"]), ("photo_7", inv["photo_7"]["pallet_empty"][0])):
    obst = B.obstacle_cells(grid, frame, [box], px)
    dm = B.dual_map(cfg, pgm, layers, obst, grid, frame, urdf["envelope_cm"], nominal)
    corners = frame.img_to_floor([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    ym = frame.floor_to_map(corners)[:, 1]
    print(f"== {photo} pallet box {box}: footprint y_map {ym.min():.3f}..{ym.max():.3f}")
    for m in (2.0, 0.0):
        L, W = nominal[0] + 2 * m, nominal[1] + 2 * m
        Lw, Ww = dm.wheel_dims(L, W)
        rows = []
        for y in np.arange(1.15, 1.32, 0.01):
            f = frame.map_to_floor([[2.93, y]])[0]
            okw, bw = dm.w.body_free(f, Lw, Ww, 90.0)
            okb, bb = dm.b.body_free(f, L, W, 90.0)
            rows.append(f"y {y:.2f}: wheel_bad_px {bw:4d} body_bad_px {bb:4d}{'  OK' if okw and okb else ''}")
        print(f"  margin {m} cm (body {L}x{W}, wheels {Lw:.1f}x{Ww:.1f}) at x_map 2.93, yaw 90, 1 cm steps (diagnostic; planner uses 2 cm cell centres):")
        print("   " + "\n   ".join(rows))
