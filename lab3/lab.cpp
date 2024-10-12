#include <iostream>
#include "mpi.h"

#define N 4
int main(int argc, char **argv)
{
    int rank;
    int total_processes;
    const int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    int *numbers = new int[N * total_processes];
    int *number_part = new int[N];

    if (rank == root)
    {
        std::cout << "Process 0 scatter data: ";
        for (int i = 0; i < N * total_processes; ++i)
        {
            numbers[i] = i + 1;
            std::cout << numbers[i] << " ";
        }
        std::cout << "\n" << std::flush;
    }

    MPI_Scatter(numbers, N, MPI_INT, number_part, N, MPI_INT, root, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i)
    {
        number_part[i] *= rank;
    }

    int sum = 0;

    for (int i = 0; i < N; ++i)
    {
        sum += number_part[i];
    }

    MPI_Gather(number_part, N, MPI_INT, numbers, N, MPI_INT, root, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == root)
    {
        std::cout << "Process 0 gathered data: ";
        for (int i = 0; i < N * total_processes; ++i)
        {
            std::cout << numbers[i] << " ";
        }
        std::cout << "\n" << std::flush;
    }

    int global_sum = 0;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Process " << rank << " sums: Local: " << sum << "; Global: " << global_sum << "\n" << std::flush;

    MPI_Finalize();
}