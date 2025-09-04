import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm 

def read_field_from_file(filename, num_halo=None):
    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset = (3 + rank) * 32 // nbits
    data = np.fromfile(
        filename,
        dtype=np.float32 if nbits == 32 else np.float64,
        count=nz * ny * nx + offset,
    )
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))


def validate_results():
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    in_field = read_field_from_file("in_field.dat")
    im1 = axs[0].imshow(
        in_field[in_field.shape[0] // 2, :, :], origin="lower", vmin=-0.1, vmax=1.1
    )
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Initial condition")

    out_field = read_field_from_file("out_field.dat")
    im2 = axs[1].imshow(
        out_field[out_field.shape[0] // 2, :, :], origin="lower", vmin=-0.1, vmax=1.1
    )
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Final result")

    plt.show()

def bar_plot_omp_runtime(times_masked, mpi_rank):
    # OMP Threads, MPI Rank const
    # times_masked: 2d runtime of OMP-MPI combinations (#OMP, #MPI)

    labels = np.arange(len(times_masked[:, mpi_rank]))
    x = labels  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(labels, times_masked[:, mpi_rank])
    ax.set_xlabel('Number of OMP Threads')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f"OMP Threads vs Runtimes, {mpi_rank} MPI Rank")

    return fig

def bar_plot_mpi_runtime(times_masked, omp_threads):
    # MPI Ranks, OMP Threads const
    # times_masked: 2d runtime of OMP-MPI combinations (#OMP, #MPI)
    
    labels = np.arange(len(times_masked[omp_threads, :]))
    x = labels  # the label locations
    width = 0.35  # the width of the bars
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    # plt.plot(runtimes[:][1])
    ax.bar(labels, times_masked[omp_threads, :])
    ax.set_xlabel('Number of MPI Ranks')
    ax.set_ylabel('Runtime (s)')
    ax.set_title(f"MPI Ranks vs Runtimes, {omp_threads} OMP Thread")
    return fig


def plot_speedup_omp_threads(times_masked, nx_len, ny_len, mpi_ranks='all'):
    T_baseline=times_masked[1][1]
    
    nthreads, nranks = times_masked.shape
    threads = np.arange(nthreads)
    if mpi_ranks == 'all':
        ranks = np.arange(nranks)  # column indices = MPI ranks
    else:
        ranks = mpi_ranks
    
    cmap = cm.bamako_r
    colors = [cmap(i / (len(ranks) - 1)) for i in range(len(ranks))]
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', 'h', '<', '>', '8', 'p']
    
    fig = plt.figure(figsize=(10, 6))
    
    for idx, (j, color) in enumerate(zip(ranks, colors)):
        runtimes_j = T_baseline/(times_masked[:, j])
        # runtimes_j=(runtimes_array[:, j])
        if np.any(runtimes_j > 0):
            marker = markers[j % len(markers)]  # Cycle through markers
            plt.plot(threads, runtimes_j, label=f"{j} MPI ranks", color=color, marker=marker)
    
    plt.xlabel("OMP Threads")
    plt.ylabel("Speedup")
    # plt.xscale("log",  base=2)
    # plt.yscale("log",  base=2)
    plt.title(f"Speedup vs OMP Threads for Different MPI Ranks [{nx_len}, {ny_len}]")
    plt.legend(title="MPI Ranks", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xlim(1,74)
    plt.tight_layout()
    return fig

def plot_heatmap(times_masked, nx_len, ny_len, mpi_ranks="all", omp_threads='all'):
    T_baseline=times_masked[1][1]
    labels = []
    times = []
    speedups= []

    nthreads, nranks = times_masked.shape
    if mpi_ranks == "all":
        nranks = range(nranks)
    else:
        nranks = mpi_ranks

    if omp_threads == "all":
        nthreads = range(nthreads)
    else:
        nthreads = omp_threads

    for thread in nthreads:
        for rank in nranks:
            runtime = times_masked[thread, rank]
            if runtime > 0.0:  # only plot entries that were filled
                labels.append([thread, rank])
                times.append(runtime)
                speedups.append(T_baseline/runtime)

    labels_array = np.array(labels)
    # threads = sorted(set(labels_array[:, 0]))
    # ranks = sorted(set(labels_array[:, 1]))

    threads = nthreads
    ranks = nranks
    
    heatmap = np.full((len(threads), len(ranks)), np.nan)
    
    for (t, r), val in zip(labels, speedups):
        i = threads.index(t)
        j = ranks.index(r)
        heatmap[i, j] = val
    heatmap=heatmap.T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.imshow(heatmap, cmap=cm.bamako, origin='lower', 
                  vmin=1,
                  aspect='auto')
    
    ax.set_ylabel('MPI Ranks')
    ax.set_xlabel('OMP Threads')
    ax.set_title(f'Speedup for domain size [{nx_len}, {ny_len}]')
    fig.colorbar(c, label='Speedup', shrink=0.8) 
    ax.set_yticks(np.arange(len(ranks)))
    ax.set_xticks(np.arange(len(threads)))
    ax.set_xticklabels(nthreads)
    ax.set_yticklabels(ranks)
    plt.tight_layout()
    return fig





































