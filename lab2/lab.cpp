#include <iostream>
#include "mpi.h"

#define N 10
int main(int argc, char **argv)
{
    int rank;
    const int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    int *numbers = new int[N];

    if (rank == root)
    {
        for (int i = 0; i < N; ++i)
        {
            numbers[i] = i + 1;
        }
        std::cout << "Process 0 broadcasting data...\n" << std::flush;
    }

    MPI_Bcast(numbers, N, MPI_INT, root, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i)
    {
        numbers[i] *= rank;
    }

    int sum = 0;
    int max = numbers[0];

    for (int i = 0; i < N; ++i)
    {
        sum += numbers[i];
        max = std::max(max, numbers[i]);
    }

    // std::cout << rank << " " << sum << " " << max << "\n" << std::flush;

    int global_sum;
    int global_max;
    
    MPI_Reduce(&sum, &global_sum, 1, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(&max, &global_max, 1, MPI_INT, MPI_MAX, root, MPI_COMM_WORLD);

    // MPI_Barrier(MPI_COMM_WORLD);

    if (rank == root) {
        std::cout << "Total sum of all elements: " << global_sum << "\n" << std::flush;
        std::cout << "Maximum element: " << global_max << "\n" << std::flush;
    }

    MPI_Finalize();
}