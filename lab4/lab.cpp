#include <iostream>
#include <chrono>
#include "mpi.h"

#define N 4

float duration(auto start_time, auto end_time)
{
    return std::chrono::duration<float>(end_time - start_time).count();
}

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
        std::cout << std::endl;
    }

    auto start_time = std::chrono::system_clock::now();

    MPI_Scatter(numbers, N, MPI_INT, number_part, N, MPI_INT, root, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i)
    {
        number_part[i] *= rank;
    }

    MPI_Gather(number_part, N, MPI_INT, numbers, N, MPI_INT, root, MPI_COMM_WORLD);

    auto end_time = std::chrono::system_clock::now();

    if (rank == root)
    {
        std::cout << "Sync data time: " << duration(start_time, end_time) << " sec.\n";
        std::cout << "Sync data: ";
        for (int i = 0; i < N * total_processes; ++i)
        {
            std::cout << numbers[i] << " ";
        }
        std::cout << std::endl;
    }

    start_time = std::chrono::system_clock::now();

    MPI_Request request;

    MPI_Iscatter(numbers, N, MPI_INT, number_part, N, MPI_INT, root, MPI_COMM_WORLD, &request);

    for (int i = 0; i < N; ++i)
    {
        number_part[i] *= rank;
    }

    MPI_Igather(number_part, N, MPI_INT, numbers, N, MPI_INT, root, MPI_COMM_WORLD, &request);

    MPI_Status status;
    MPI_Wait(&request, &status);

    end_time = std::chrono::system_clock::now();

    if (rank == root)
    {
        std::cout << "Async data time: " << duration(start_time, end_time) << " sec.\n";
        std::cout << "Async data: ";
        for (int i = 0; i < N * total_processes; ++i)
        {
            std::cout << numbers[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
}