import copy
import time
import numpy as np
import open3d as o3d
import teaserpp_python

NOISE_BOUND = 0.05
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 10


def get_angular_error(R_exp, R_est):
    """
    Calculate angular error
    """
    return abs(np.arccos(min(max(((np.matmul(R_exp.T, R_est)).trace() - 1) / 2, -1.0), 1.0)));


if __name__ == "__main__":
    print("==================================================")
    print("        TEASER++ Python registration example      ")
    print("==================================================")

    # Load bunny ply file
    src_cloud = o3d.io.read_point_cloud("/home/vince/Experiments/idepth/results_3d/2019-01-25_17-45-17_part0001/left.ply")
    dst_cloud = o3d.io.read_point_cloud("/home/vince/Experiments/idepth/results_3d/2019-01-25_17-45-17_part0001/left2.ply")
    src = np.transpose(np.asarray(src_cloud.points))
    dst = np.transpose(np.asarray(dst_cloud.points))
    NOISE_BOUND = max(np.std(src[2])*2, np.std(dst[2])*2)
    print("NOISE_BOUND: ")
    print(NOISE_BOUND)

    N_src = src.shape[1]
    N_dst = dst.shape[1]

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()
    solver.solve(src, dst)
    end = time.time()

    solution = solver.getSolution()

    print("=====================================")
    print("          TEASER++ Results           ")
    print("=====================================")


    print("Estimated rotation: ")
    print(solution.rotation)

    print("Estimated translation: ")
    print(solution.translation)


    print("Number of src correspondences: ", N_src)
    print("Number of dst correspondences: ", N_dst)
    print("Time taken (s): ", end - start)

