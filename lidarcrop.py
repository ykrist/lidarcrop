#!/usr/bin/env python
import laspy
import sklearn.cluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plyfile
import matplotlib.patches
import scipy.spatial
import yaml
import argparse
import sys
import os

from numpy.lib.recfunctions import append_fields


_DUMMY_FILENAME = "__NO_FILENAME__"

def trim_tails(z, tol=0):
    # Finds the start and end indices from the 1-D array *z* for which *z[start:end]* excludes ascent/descent sequence.
    start_index = 0
    tol = abs(tol)
    error = Exception("Tolerence is too high.")
    while z[start_index + 1] - z[start_index] > -tol:
        start_index += 1
        if start_index == len(z) - 1:
            raise error
    end_index = len(z) - 1
    while z[end_index - 1] - z[end_index] > -tol:
        end_index -= 1
        if end_index == 0:
            return error
    assert start_index < end_index
    return (start_index, end_index)


def config_dict_to_cl_args(config):
    args = []
    switches = ["debug", "plot", "txt-summary", "yaml-summary"]
    optargs = ["name", 'tol', 'min-range', 'max-range', "fov-angle", "polygon-radius", "polygon-vertices",
               "sor-numpoints", "sor-nsigma", "alt-radius"]
    posargs = ["plyfile", "trajfile", "lasfile"]

    for p in posargs:
            val = config.pop(p)
            if val is not None:
                args.append(str(val))

    for a in optargs:
            val = config.pop(a)
            if val is not None:
                args.extend(["--{}".format(a), str(val)])

    for s in switches:
            val = config.pop(s)
            if val:
                args.append("--{}".format(s))


    assert len(config) == 0, "Error parsing config file, parameters not handled: {}".format(list(config.keys()))

    return args


def get_args():
    # Parse command-line options.
    args = argparse.ArgumentParser()
    args.add_argument("plyfile", type=str, help="Path to input .ply file (required)")
    args.add_argument("trajfile", type=str, help="Path to input trajectory .txt file. (required)")
    args.add_argument("lasfile", type=str, nargs='?', default=None,
                      help="Path to input .las file (if laszip is in the PATH, .laz will work " \
                           "too). If none is supplied, there will be no LAS output file.")
    args.add_argument("-n", "--name", type=str, default=os.getcwd() + "/out",
                      help="Name of output (used for output .las and trajectory file).  May be prefixed with path. Will "
                           "append NAME with '.las' and '_traj.txt' for point-cloud and trajectory files, respectively.")
    args.add_argument("--tol", type=float, help="Tolerance when trimming ascent/descent from trajectory (default is 0)",
                      default=0.0)
    args.add_argument("--min-range", type=float, help="Keep only points with range > min_range",
                      default=0.0)
    args.add_argument("--max-range", type=float, help="Keep only points with range < max_range.  Set to 'inf' to " \
                                                      "disable.",
                      default=float('inf'))
    args.add_argument("--fov-angle", type=float, help="Restrict points by FOV angle (degrees).  This angle is measured "
                                                      "from the negative z-axis. Set to <= 0 to disable. (default is 0)",
                      default=0)
    args.add_argument("--alt-radius", type=float, help="Radius of points to use when calculating altitude (default "
                                                       "is 0.2)",
                      default=0.2)
    args.add_argument("--polygon-vertices", type=int, help="Number of polygon sides to use when creating the bounding "
                                                           "hull.",
                      default=50)
    args.add_argument("--polygon-radius", type=float, help="Length of polygon sides to use when creating the "
                                                           "bounding hull.",
                      default=5)
    args.add_argument("--sor-nsigma", type=float, help="Significant Outlier Removal: Number of STD above the mean a "
                                                       "point should be before it's removed (default is 2)",
                      default=2)
    args.add_argument("--sor-numpoints", type=int, help="Significant Outlier Removal: Number of points to use in "
                                                        "computing the mean distance to neighbours (default is 8)",
                      default=8)
    args.add_argument("-p", "--plot", action="store_true", help="Plot the trajectory and bounding hull.")
    args.add_argument("-d", "--debug", action="store_true", help="Debug mode: do not create any files.")
    args.add_argument("-o", "--txt-summary", action="store_true", help="Output a summary of results to a file named"
                                                                       " NAME.txt.")
    args.add_argument("-y", "--yaml-summary", action="store_true", help="Output a summary of results to a file named"
                                                                        " NAME.yaml.")
    args.epilog = "Parameters specified in the config.yaml file will be used if they are present.  However, " \
                  "if both parameter is given in both config.yaml and in the command line arguments, the command-line " \
                  "specification will override those in the config.yaml file."

    config_filename = "config.yaml"
    if os.path.exists(config_filename):
        with open("config.yaml", "r") as f:
            config_dict = yaml.load(f)

        if len(sys.argv) == 1:
            # No CL args
            args = args.parse_args(config_dict_to_cl_args(config_dict))

        elif len(sys.argv) >= 3:
            # CL args supplied
            num_pos_cl_args = len(list(filter(lambda x : x[0] != '-', sys.argv[1:])))
            if num_pos_cl_args == 2:
                config_dict["plyfile"] = None
                config_dict["trajfile"] = None
            elif num_pos_cl_args == 3:
                config_dict["plyfile"] = None
                config_dict["trajfile"] = None
                config_dict["lasfile"] = None
            else:
                args.print_usage()
                exit(1)

            conf_args = config_dict_to_cl_args(config_dict)
            cl_args = sys.argv[1:]
            for a in cl_args:
                if a in args._option_string_actions.keys():
                    nargs = args._option_string_actions[a].nargs
                    long_name = args._option_string_actions[a].option_strings[-1]
                    if long_name in conf_args:
                        i = conf_args.index(a)
                        conf_args.pop(i)
                        if nargs is None:
                            conf_args.pop(i)
            args = args.parse_args(cl_args + conf_args)

        else:
            args.print_usage()
            exit(1)
    else:
        if len(sys.argv) < 2:
            args.print_usage()
            exit(1)

        else:
            args = args.parse_args(sys.argv[1:])

    args.lasfile = os.path.abspath(args.lasfile) if args.lasfile is not None else None
    args.trajfile = os.path.abspath(args.trajfile)
    args.plyfile = os.path.abspath(args.plyfile)
    args.name = os.path.abspath(args.name)
    return args


def sor_filter(points, args):
    kdtree = scipy.spatial.cKDTree(points)
    mean_distances, _ = kdtree.query(points, k=args.sor_numpoints + 1, n_jobs=-1)
    mean_distances = mean_distances[:, 1:].mean(axis=1)
    mean = mean_distances.mean()
    std = mean_distances.std()
    return mean_distances < mean + args.sor_nsigma * std


def get_bool_mask(index_arr, length):
    mask = np.zeros(length, dtype="bool")
    mask[index_arr.astype("uint")] = True
    return mask


def get_convex_hull(points, trajectory, args):
    polygon_radius = args.polygon_radius
    if polygon_radius < 0:
        convex_hull = scipy.spatial.ConvexHull(points[:, [1, 2]])
    elif polygon_radius == 0:
        # Use flight trajectory to generate bounding polygon
        convex_hull = scipy.spatial.ConvexHull(trajectory[:, [1, 2]])
    else:
        # Add polygons to the x-y flight path to push the flight path out
        polygon_sides = args.polygon_vertices
        polygon_frequency = 100
        polygon_centers = trajectory[::polygon_frequency, [1, 2]].copy()
        num_polygons = polygon_centers.shape[0]
        polygon_vertices = np.zeros((num_polygons * polygon_sides, 2))
        for i in range(num_polygons):
            angles = np.linspace(0, 2 * np.pi, polygon_sides) + np.random.rand() * 2 * np.pi
            xvals = polygon_radius * np.cos(angles) + polygon_centers[i, 0]
            yvals = polygon_radius * np.sin(angles) + polygon_centers[i, 1]
            polygon_vertices[i * polygon_sides:(i + 1) * polygon_sides] = np.stack([xvals, yvals], axis=-1)

        # Build bounding polygon
        convex_hull = scipy.spatial.ConvexHull(polygon_vertices)
    return convex_hull


def calculate_angles(points, traj_points):
    angles = np.full(points.shape[0], np.NaN)
    # Bin dividers
    div = traj_points[:-1, 0] + np.diff(traj_points[:, 0]) * 0.5
    div = np.concatenate([[-np.inf], div, [np.inf]])

    num_points = len(points)
    current_index = 0

    for i in range(len(div) - 1):
        start_index = current_index
        while current_index < num_points and div[i] < points[current_index, 0] <= div[i + 1]:
            current_index += 1
        end_index = current_index

        if start_index == end_index:
            continue

        xyz_points = points[start_index:end_index, [1, 2, 3]] - traj_points[i, [1, 2, 3]]
        xyz_points /= np.linalg.norm(xyz_points, axis=1, keepdims=True)
        angles[start_index:end_index] = np.arccos(-xyz_points[:, 2])

    return np.rad2deg(angles)


def calculate_altitude(points, traj_points, args):
    alt = np.zeros(len(traj_points))
    points_kdtree = scipy.spatial.cKDTree(points[:, [1, 2]])
    for i in range(len(traj_points)):
        indices = points_kdtree.query_ball_point(traj_points[i, [1, 2]], r=args.alt_radius)
        alt[i] = traj_points[i, 3] - points[indices, 3].mean()
    return alt


def calculate_distance_velocity(traj_points):
    distance = np.linalg.norm(traj_points[1:, 1:] - traj_points[:-1, 1:], axis=1)
    delta_t = np.diff(traj_points[:, 0])
    velocity = distance / delta_t
    return distance, velocity


def create_summary_str(summary):
    s = ""
    s += "Number of input points: {:,}".format(summary["input-point-count"]) + "\n"
    s += "Number of points removed:\n\tTime: {:,}\n\tClipping: {:,}\n\tRange: {:,}\n\tSOR: {:,}" \
         "\n\tFOV: {:,}\n\tTotal: {:,}\n".format(
        summary["points-removed-time"], summary["points-removed-clip"], summary["points-removed-range"],
        summary["points-removed-sor"], summary["points-removed-fov"], summary["input-point-count"] - \
                                                                      summary["output-point-count"]
    )
    s += "Number of points kept: {:,}\n".format(summary["output-point-count"])
    s += "Max range: {:.3g} m\nMin range: {:.3g} m\nMean range: {:.3g} m\nRange STDEV: {:.3g} m\n".format(
        summary["range-max"], summary["range-min"], summary["range-mean"], summary["range-std"]
    )
    s += "Start time of real flight path: {:.3g} s\n".format(summary["start-time"])
    s += "End time of real flight path: {:.3g} s\n".format(summary["end-time"])
    s += "Total flight time for real flight path: {:.3g} s\n".format(summary["total-time"])
    s += "Total distance travelled for real flight path: {:.3g} m\n".format(summary["traj-total-distance"])
    s += "Trajectory pattern: {:.3g} m^-1\n".format(summary["traj-pattern"])
    s += "Trajectory loiter: {:.3g} s/m\n".format(summary["traj-loiter"])
    s += "Mean altitude of real flight path: {:.3g} m\n".format(summary["traj-mean-alt"])
    s += "Standard deviation of real flight path altitude : {:.3g} m\n".format(summary["traj-std-alt"])
    s += "Mean speed of real flight path: {:.3g} m/s\n".format(summary["traj-mean-speed"])
    s += "Standard deviation of real flight path speed : {:.3g} m/s\n".format(summary["traj-std-speed"])
    s += "Final x-y area of point-cloud extent: {:.3g} m^2\n".format(summary["clipped-area"])
    s += "Spatial point cloud density (2.5D): {:.3g} pts/m^2\n".format(summary["point-density-2d"])
    s += "Spatial mean point distance (2.5D): {:.3g} m\n".format(summary["point-mean-distance-2d"])
    s += "Sampling effort variable: {:.3g} s/m^4\n".format(summary["traj-sev"])
    s += "Effective sampling rate: {:.3g} pts/s\n".format(summary["traj-esr"])
    s += "Effective density rate: {:.3g} pts/(s m^2)\n".format(summary["traj-edr"])
    return s


# -------------------------------------- MAIN SCRIPT -------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    summary = {
        "input-point-count": 0,
        "output-point-count": 0,
        "point-density-2d": "N/A",
        "point-mean-distance-2d": "N/A",
        "points-removed-clip": 0,
        "points-removed-sor": 0,
        "points-removed-range": 0,
        "points-removed-time": 0,
        "points-removed-fov": 0,
        "start-time": "N/A",
        "end-time": "N/A",
        "clipped-area": 0,
        "range-min": "N/A",
        "range-max": "N/A",
        "fov-horizontal-length": "N/A",
        "traj-mean-alt": "N/A",
        "traj-std-alt": "N/A",
    }
    # Read input files
    traj_df = pd.read_csv(args.trajfile, sep=" ")
    traj_df.drop(columns="userfields", inplace=True, errors='ignore')
    with open(args.plyfile, "rb") as f:
        ply_data = plyfile.PlyData.read(f)

    # Grab points from PLY file
    data_points = np.array([
        ply_data['vertex']['time'],
        ply_data['vertex']['x'],
        ply_data['vertex']['y'],
        ply_data['vertex']['z'],
        ply_data['vertex']['range'],
        np.arange(len(ply_data['vertex'].data))  # indices
    ]).transpose()
    summary["input-point-count"] = int(data_points.shape[0])

    # Get trajectory data (numpy array with shape (num_points, 4) where each row is [t, x, y, z]
    trajectory = traj_df.loc[:, ["%time", "x", "y", "z"]].values.copy()

    if args.tol >= 0:
        print("Trimming trajectory...")
        # Cluster the points into two means (Marked as red and blue)
        clusterer = sklearn.cluster.KMeans(2, n_jobs=-1)
        cluster_index = clusterer.fit_predict(trajectory[:, [3]])
        means = clusterer.cluster_centers_

        # Trim the ascent/descent tails (output from this step is shown in black)
        mask = cluster_index == means.argmax()
        filtered_trajectory = trajectory[mask, :]
        start, end = trim_tails(filtered_trajectory[:, 3], tol=args.tol)
        filtered_trajectory = filtered_trajectory[start:end]

        # Get start and end times to filter points by
        start_time = filtered_trajectory[0, 0]
        end_time = filtered_trajectory[-1, 0]
        mask = np.logical_and(data_points[:, 0] <= end_time, start_time <= data_points[:, 0])
        summary["points-removed-time"] = int(np.sum(1 - mask))
        data_points = data_points[mask]
    else:
        filtered_trajectory = trajectory
        start_time = filtered_trajectory[0, 0]
        end_time = filtered_trajectory[-1, 0]
        means = []

    summary["start-time"] = float(start_time)
    summary["end-time"] = float(end_time)
    summary["total-time"] = summary["end-time"] - summary["start-time"]
    # Clip using a polygonal boundary
    print("Calculating bounding polygon...")
    convex_hull = get_convex_hull(data_points, filtered_trajectory, args)
    if args.polygon_radius >= 0:
        # Cut out points not in polygon
        print("Filtering points not in bounding polygon...")
        triangulation = scipy.spatial.Delaunay(convex_hull.points[convex_hull.vertices].copy())
        mask = triangulation.find_simplex(data_points[:, [1, 2]]) >= 0
        summary["points-removed-clip"] = int(np.sum(1 - mask))
        data_points = data_points[mask]
        del triangulation

    summary["clipped-area"] = float(convex_hull.volume)

    # Range filtering
    if args.max_range != np.inf or args.min_range == 0:
        print("Filtering points based on range"
              " (keeping points with range in the interval [{},{}] )...".format(args.min_range, args.max_range))
        mask = np.logical_and(data_points[:, 4] <= args.max_range, args.min_range <= data_points[:, 4])
        summary["points-removed-range"] = int(np.sum(1 - mask))
        data_points = data_points[mask]

    # Calculate angles (these will be added as point properties in the the output PLY file)
    print("Calculating angles...")
    angles = calculate_angles(data_points, filtered_trajectory)
    if args.fov_angle > 0:
        print("Filtering points based on FOV angle (FOV is set to {} degrees)...".format(args.fov_angle))
        mask = angles <= args.fov_angle
        summary["points-removed-fov"] = int(np.sum(1 - mask))
        angles = angles[mask]
        data_points = data_points[mask]

    # Significant outlier removal
    if args.sor_numpoints > 0:
        print("Filtering points using SOR filter...")
        mask = sor_filter(data_points[:, [1, 2, 3]].copy(), args)
        summary["points-removed-sor"] = int(np.sum(1 - mask))
        data_points = data_points[mask]
        angles = angles[mask]

    # Calculate the true altitude (this will be added to the output trajectory file)
    print("Calculating true altitude...")
    true_altitude = calculate_altitude(data_points, filtered_trajectory, args)
    summary["traj-mean-alt"] = float(true_altitude.mean())
    summary["traj-std-alt"] = float(true_altitude.std())

    distance, velocity = calculate_distance_velocity(filtered_trajectory)
    summary["traj-mean-speed"] = float(velocity.mean())
    summary["traj-std-speed"] = float(velocity.std())
    summary["traj-total-distance"] = float(distance.sum())
    summary["traj-pattern"] = summary["traj-total-distance"] / summary["clipped-area"]
    summary["traj-loiter"] = summary["traj-pattern"] * summary["total-time"]
    summary["range-mean"] = float(data_points[:, 4].mean())
    summary["range-std"] = float(data_points[:, 4].std())
    summary["range-max"] = float(data_points[:, 4].max())
    summary["range-min"] = float(data_points[:, 4].min())
    summary["traj-sev"] = summary["traj-loiter"] / summary["range-mean"]
    summary["traj-esr"] = float(len(data_points) / summary["total-time"])
    summary["traj-edr"] = float(summary["traj-esr"] / convex_hull.volume)

    # Calculate point cloud density
    point_density_2d = len(data_points) / convex_hull.volume
    summary["point-density-2d"] = float(point_density_2d)
    summary["point-mean-distance-2d"] = float(1 / np.sqrt(point_density_2d))
    summary["output-point-count"] = summary["input-point-count"] - (summary["points-removed-range"] +
                                                                    summary["points-removed-clip"] +
                                                                    summary["points-removed-sor"] +
                                                                    summary["points-removed-fov"] +
                                                                    summary["points-removed-time"])
    summary_string = create_summary_str(summary)

    # Write the output to file
    if not args.debug:
        point_mask = get_bool_mask(data_points[:, -1], summary["input-point-count"])
        assert np.sum(point_mask) == data_points.shape[0]
        del data_points

        # Create the new las file
        if args.lasfile is not None:
            print("Writing LAS file...")
            in_las_file = laspy.file.File(args.lasfile, mode="r")
            las_points = in_las_file.points.copy()
            assert len(las_points) == len(point_mask), "Number of points in .las file must match number of points in " \
                                                       ".ply file."
            out_las_file = laspy.file.File(args.name + ".las", mode="w", header=in_las_file.header)
            out_las_file.points = las_points[point_mask]
            out_las_file.close()
            in_las_file.close()

        print("Writing PLY file...")
        byte_order = ply_data.byte_order
        ply_data = np.array(ply_data["vertex"].data[point_mask])
        ply_data = append_fields(ply_data, "angle", angles).data
        with open(args.name + ".ply", "wb") as f:
            el = plyfile.PlyElement.describe(ply_data, 'vertex')
            plyfile.PlyData([el], byte_order=byte_order).write(args.name + ".ply")

        print("Writing trajectory file...")
        # Create the new trajectory file
        traj_mask = np.logical_and(start_time <= traj_df.loc[:, "%time"], traj_df.loc[:, "%time"] <= end_time)
        traj_df = traj_df.loc[traj_mask, :]
        traj_df["alt"] = true_altitude
        traj_df.to_csv(args.name + "_traj.txt", sep=" ", index=False)

        # Log output to file
        if args.txt_summary:
            print("Writing summary to txt file...")
            with open(args.name + "_summary.txt", "w+") as f:
                f.write(summary_string)

        if args.yaml_summary:
            print("Writing summary to yaml file...")
            with open(args.name + "_summary.yaml", "w+") as f:
                yaml.dump(summary, f, default_flow_style=False)

    # Plot the elevation over time, split into the clusters, along with the final filtered trajectory (black)
    if args.plot:
        # Restrict the number of points actually plotted to stop matplotlib from shitting itself
        num_points = 500
        restrict_num_points = lambda arr: arr[::int(len(arr) / (num_points + 0.0))]

        colormap = ['r', 'b', 'k', 'm', 'g']
        fig1 = plt.figure(0, figsize=(10, 6), dpi=100)

        ax1a = fig1.add_subplot(1, 2, 1)

        if args.tol >= 0:
            for i, v in enumerate(means):
                ax1a.axhline(v, c=colormap[i])
            ax1a.scatter(restrict_num_points(trajectory)[:, 0], restrict_num_points(trajectory)[:, 3],
                         c=list(map(lambda x: colormap[x], restrict_num_points(cluster_index))))
        ax1a.scatter(restrict_num_points(filtered_trajectory)[:, 0], restrict_num_points(filtered_trajectory)[:, 3],
                     c="k")
        ax1a.scatter(restrict_num_points(filtered_trajectory)[:, 0], restrict_num_points(true_altitude),
                     c="g")
        ax1a.set_ylabel("z")
        ax1a.set_xlabel("Time")
        ax1a.set_title("Elevation")

        ax1b = fig1.add_subplot(1, 2, 2)
        ax1b.scatter(restrict_num_points(trajectory)[:, 1], restrict_num_points(trajectory)[:, 2],
                     c=list(map(lambda x: colormap[x], restrict_num_points(cluster_index))))
        ax1b.scatter(restrict_num_points(filtered_trajectory)[:, 1], restrict_num_points(filtered_trajectory)[:, 2],
                     c='k')
        ax1b.set_xlabel("x")
        ax1b.set_ylabel("y")
        ax1b.add_patch(matplotlib.patches.Polygon(convex_hull.points[convex_hull.vertices].copy(), facecolor='none',
                                                  edgecolor="k", linestyle='-'))
        ax1b.autoscale()
        ax1b.set_title("Top-down")
        ax1b.set_aspect('equal')

        fig1.savefig(args.name + "_plot.png")

    print("Done.\n\n" + 35 * "-" + " SUMMARY " + 35 * "-" + "\n" + summary_string)
