# %% Import packages

import matplotlib.pyplot as plt
import numpy as np

from pgf_kernel_experiments.experiments.cox_process.kernel_comparison.setting01.set_env import (
    data_paths, dpi, num_classes, num_runs, output_paths
)
from pgf_kernel_experiments.plots import set_axes_equal

# %% Generate and save plots of data

verbose = True
if verbose:
    num_run_digits = len(str(num_runs))
    msg = 'Plotting predictions of run {:'+str(num_run_digits)+'d}/{:'+str(num_run_digits)+'d}...'

xyz_lim = 0.63

title_fontsize = 15
colorbar_fontsize = 11

titles = [
    'Trigonometric function', 'Noiseless training points', '',
    'Test points', 'PGF kernel', 'RBF kernel',
    'Matern kernel', 'Periodic kernel', 'Spectral kernel'
]

for i in range(num_runs):
    # If verbose, state run number

    if verbose:
        print(msg.format(i+1, num_runs))

    # Load data

    data = np.loadtxt(
        data_paths[i].joinpath('data.csv'),
        delimiter=',',
        skiprows=1
    )

    grid = data[:, 2:5]
    x = data[:, 2]
    y = data[:, 3]
    z = data[:, 4]
    v = data[:, 5]

    dims = np.loadtxt(data_paths[i].joinpath('dims.csv'), dtype='int')

    train_ids = np.loadtxt(data_paths[i].joinpath('train_ids.csv'), dtype='int')

    test_ids = np.loadtxt(data_paths[i].joinpath('test_ids.csv'), dtype='int')
    if num_test is not None:
        test_ids = test_ids[:num_test]
    test_ids.sort()

    # Load predictions

    predictions = np.loadtxt(
        output_paths[i].joinpath('predictions.csv'),
        delimiter=',',
        skiprows=1
    )

    # Get training data

    train_pos = grid[train_ids, :]
    train_output = v[train_ids]

    # Get test data

    test_pos = grid[test_ids, :]
    test_output = v[test_ids]

    # Reshape data for plotting

    x_plot = x.reshape(dims[0], dims[1], order='C')
    x_plot = np.vstack([x_plot, x_plot[0, :]])

    y_plot = y.reshape(dims[0], dims[1], order='C')
    y_plot = np.vstack([y_plot, y_plot[0, :]])

    z_plot = z.reshape(dims[0], dims[1], order='C')
    z_plot = np.vstack([z_plot, z_plot[0, :]])

    v_plot = v.reshape(dims[0], dims[1], order='C')
    v_plot = np.vstack([v_plot, v_plot[0, :]])

    # Generate plot points for training data

    num_samples = dims[0] * dims[1]
    ids = np.arange(num_samples)

    non_train_ids = np.array(list(set(ids).difference(set(train_ids))))
    non_train_ids.sort()

    train_v_plot = v.flatten()
    train_v_plot[non_train_ids] = np.nan
    train_v_plot = train_v_plot.reshape(dims[0], dims[1], order='C')
    train_v_plot = np.vstack([train_v_plot, train_v_plot[0, :]])

    # Generate plot points for test data

    non_test_ids = np.array(list(set(ids).difference(set(test_ids))))
    non_test_ids.sort()

    test_v_plot = v.flatten()
    test_v_plot[non_test_ids] = np.nan
    test_v_plot = test_v_plot.reshape(dims[0], dims[1], order='C')
    test_v_plot = np.vstack([test_v_plot, test_v_plot[0, :]])

    # Generate plot points for GP predictions based on PGF kernel

    pgf_gp_v_plot = np.full_like(v, np.nan).flatten()
    pgf_gp_v_plot[test_ids] = predictions[:, 0]
    pgf_gp_v_plot = pgf_gp_v_plot.reshape(dims[0], dims[1], order='C')
    pgf_gp_v_plot = np.vstack([pgf_gp_v_plot, pgf_gp_v_plot[0, :]])

    # Generate plot points for GP predictions based on RBF kernel

    rbf_gp_v_plot = np.full_like(v, np.nan).flatten()
    rbf_gp_v_plot[test_ids] = predictions[:, 1]
    rbf_gp_v_plot = rbf_gp_v_plot.reshape(dims[0], dims[1], order='C')
    rbf_gp_v_plot = np.vstack([rbf_gp_v_plot, rbf_gp_v_plot[0, :]])

    # Generate plot points for GP predictions based on Matern kernel

    matern_gp_v_plot = np.full_like(v, np.nan).flatten()
    matern_gp_v_plot[test_ids] = predictions[:, 2]
    matern_gp_v_plot = matern_gp_v_plot.reshape(dims[0], dims[1], order='C')
    matern_gp_v_plot = np.vstack([matern_gp_v_plot, matern_gp_v_plot[0, :]])

    # Generate plot points for GP predictions based on periodic kernel

    periodic_gp_v_plot = np.full_like(v, np.nan).flatten()
    periodic_gp_v_plot[test_ids] = predictions[:, 3]
    periodic_gp_v_plot = periodic_gp_v_plot.reshape(dims[0], dims[1], order='C')
    periodic_gp_v_plot = np.vstack([periodic_gp_v_plot, periodic_gp_v_plot[0, :]])

    # Generate plot points for GP predictions based on spectral kernel

    spectral_gp_v_plot = np.full_like(v, np.nan).flatten()
    spectral_gp_v_plot[test_ids] = predictions[:, 4]
    spectral_gp_v_plot = spectral_gp_v_plot.reshape(dims[0], dims[1], order='C')
    spectral_gp_v_plot = np.vstack([spectral_gp_v_plot, spectral_gp_v_plot[0, :]])

    # Plot data, including separate training and test data (adding color map)

    fig = plt.figure(figsize=[14, 18])

    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_adjust.html
    # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots
    # https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/

    fig.subplots_adjust(
        left=-0.1,
        bottom=0.0,
        right=1.0,
        top=1.0,
        wspace=-0.02,
        hspace=-0.45
    )

    ax1 = fig.add_subplot(3, 3, 1, projection='3d')

    norm = plt.Normalize()

    # https://stackoverflow.com/questions/2578752/how-can-i-plot-nan-values-as-a-special-color-with-imshow-in-matplotlib

    cmap = plt.cm.jet
    cmap.set_bad('white')

    ax1.view_init(elev=30, azim=-50, roll=10)
    # ax1.view_init(elev=30, azim=-60, roll=10) # default

    # https://github.com/matplotlib/matplotlib/issues/14647

    ax1.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(v_plot)), edgecolor='none')

    ax1.set_title(titles[0], fontsize=title_fontsize)

    ax1.set_box_aspect([1, 1, 1])

    # ax1.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax1)

    ax1.grid(False)
    ax1.axis('off')

    ax1.set_xlim(-xyz_lim, xyz_lim)
    ax1.set_ylim(-xyz_lim, xyz_lim)
    ax1.set_zlim(-xyz_lim, xyz_lim)

    ax2 = fig.add_subplot(3, 3, 2, projection='3d')

    ax2.view_init(elev=30, azim=-50, roll=10)
    # ax2.view_init(elev=30, azim=-60, roll=10) # default

    ax2.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(train_v_plot)), edgecolor='none')

    ax2.set_title(titles[1], fontsize=title_fontsize)

    ax2.set_box_aspect([1, 1, 1])

    # ax2.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax2)

    ax2.grid(False)
    ax2.axis('off')

    ax2.set_xlim(-xyz_lim, xyz_lim)
    ax2.set_ylim(-xyz_lim, xyz_lim)
    ax2.set_zlim(-xyz_lim, xyz_lim)

    ax4 = fig.add_subplot(3, 3, 4, projection='3d')

    ax4.view_init(elev=30, azim=-50, roll=10)
    # ax4.view_init(elev=30, azim=-60, roll=10) # default

    ax4.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(test_v_plot)), edgecolor='none')

    ax4.set_title(titles[3], fontsize=title_fontsize)

    ax4.set_box_aspect([1, 1, 1])

    # ax4.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax4)

    ax4.grid(False)
    ax4.axis('off')

    ax4.set_xlim(-xyz_lim, xyz_lim)
    ax4.set_ylim(-xyz_lim, xyz_lim)
    ax4.set_zlim(-xyz_lim, xyz_lim)

    ax5 = fig.add_subplot(3, 3, 5, projection='3d')

    ax5.view_init(elev=30, azim=-50, roll=10)
    # ax5.view_init(elev=30, azim=-60, roll=10) # default

    ax5.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(pgf_gp_v_plot)), edgecolor='none')

    ax5.set_title(titles[4], fontsize=title_fontsize)

    ax5.set_box_aspect([1, 1, 1])

    # ax5.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax5)

    ax5.grid(False)
    ax5.axis('off')

    ax5.set_xlim(-xyz_lim, xyz_lim)
    ax5.set_ylim(-xyz_lim, xyz_lim)
    ax5.set_zlim(-xyz_lim, xyz_lim)

    ax6 = fig.add_subplot(3, 3, 6, projection='3d')

    ax6.view_init(elev=30, azim=-50, roll=10)
    # ax6.view_init(elev=30, azim=-60, roll=10) # default

    ax6.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(rbf_gp_v_plot)), edgecolor='none')

    ax6.set_title(titles[5], fontsize=title_fontsize)

    ax6.set_box_aspect([1, 1, 1])

    # ax6.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax6)

    ax6.grid(False)
    ax6.axis('off')

    ax6.set_xlim(-xyz_lim, xyz_lim)
    ax6.set_ylim(-xyz_lim, xyz_lim)
    ax6.set_zlim(-xyz_lim, xyz_lim)

    ax7 = fig.add_subplot(3, 3, 7, projection='3d')

    ax7.view_init(elev=30, azim=-50, roll=10)
    # ax7.view_init(elev=30, azim=-60, roll=10) # default

    ax7.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(matern_gp_v_plot)), edgecolor='none')

    ax7.set_title(titles[6], fontsize=title_fontsize)

    ax7.set_box_aspect([1, 1, 1])

    # ax7.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax7)

    ax7.grid(False)
    ax7.axis('off')

    ax7.set_xlim(-xyz_lim, xyz_lim)
    ax7.set_ylim(-xyz_lim, xyz_lim)
    ax7.set_zlim(-xyz_lim, xyz_lim)

    ax8 = fig.add_subplot(3, 3, 8, projection='3d')

    ax8.view_init(elev=30, azim=-50, roll=10)
    # ax8.view_init(elev=30, azim=-60, roll=10) # default

    ax8.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(periodic_gp_v_plot)), edgecolor='none')

    ax8.set_title(titles[7], fontsize=title_fontsize)

    ax8.set_box_aspect([1, 1, 1])

    # ax8.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax8)

    ax8.grid(False)
    ax8.axis('off')

    ax8.set_xlim(-xyz_lim, xyz_lim)
    ax8.set_ylim(-xyz_lim, xyz_lim)
    ax8.set_zlim(-xyz_lim, xyz_lim)

    ax9 = fig.add_subplot(3, 3, 9, projection='3d')

    ax9.view_init(elev=30, azim=-50, roll=10)
    # ax9.view_init(elev=30, azim=-60, roll=10) # default

    ax9.plot_surface(x_plot, y_plot, z_plot, cstride=1, rstride=1, facecolors=cmap(norm(spectral_gp_v_plot)), edgecolor='none')

    ax9.set_title(titles[8], fontsize=title_fontsize)

    ax9.set_box_aspect([1, 1, 1])

    # ax9.set_proj_type('ortho') # default is perspective

    set_axes_equal(ax9)

    ax9.grid(False)
    ax9.axis('off')

    ax9.set_xlim(-xyz_lim, xyz_lim)
    ax9.set_ylim(-xyz_lim, xyz_lim)
    ax9.set_zlim(-xyz_lim, xyz_lim)

    # https://www.geeksforgeeks.org/set-matplotlib-colorbar-size-to-match-graph/

    fig.subplots_adjust(bottom=0.0, right=0.80, top=1.0)

    # cax = fig.add_axes([0.80, 0.2, 0.01, 0.6])
    cax = fig.add_axes([0.80, 0.3, 0.01, 0.4])

    # https://stackoverflow.com/questions/33443334/how-to-decrease-colorbar-width-in-matplotlib
    # https://matplotlib.org/stable/api/cm_api.html#matplotlib.cm.ScalarMappable

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet), cax=cax) # , aspect=30)

    # https://stackoverflow.com/questions/69435068/change-colorbar-limit-for-changing-scale-with-matplotlib-3-3

    cb.mappable.set_clim(-a, a)

    # https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/

    cb_tick_points = np.arange(-a, a+0.5*a, 0.5*a)
    cb.set_ticks(cb_tick_points)
    cb.set_ticklabels(cb_tick_points)

    # https://www.tutorialspoint.com/how-do-i-change-the-font-size-of-ticks-of-matplotlib-pyplot-colorbar-colorbarbase

    cb.ax.tick_params(labelsize=colorbar_fontsize)

    # plt.show()

    # Save plot

    plt.savefig(
        output_paths[i].joinpath('predictions.pdf'),
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.1
    )

    plt.close(fig)
