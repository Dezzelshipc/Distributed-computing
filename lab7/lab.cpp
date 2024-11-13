#include <iostream>
#include <chrono>
#include <iomanip>
#include "mpi.h"
#include <random>

int N = 1e9;

double duration(auto start_time, auto end_time)
{
    return std::chrono::duration<double>(end_time - start_time).count();
}

void print_array(int *array, int len = N) {
    if (len <= 20) {
        for (int i = 0; i < len; ++i) {
            std::cout << array[i] << ' ';
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv)
{
    int rank;
    int total_processes;
    const int root = 0;

    if (argc > 1)
    {
        N = (int)std::atol(argv[1]);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    int *array = new int[N];

    if (rank == root)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> uniform(0, N);

        for (int i = 0; i < N; ++i)
        {
            array[i] = uniform(gen);
        }
        print_array(array);

        auto start_time = std::chrono::system_clock::now();

        int max = -1;

        for (int i = 0; i < N; ++i)
        {
            max = std::max(max, array[i]);
        }

        auto end_time = std::chrono::system_clock::now();

        std::cout << "Non-parallel max: " << duration(start_time, end_time) << " sec." << std::endl;
        std::cout << max << std::endl;
    }

    int max = -1;

    int numbers_per_process = N / total_processes;
    if (N % total_processes != 0)
    {
        numbers_per_process++;
    }

    int counts[total_processes];
    int displ[total_processes];

    for (int i = 0; i < total_processes; ++i)
    {
        int s = numbers_per_process * i;
        int e = std::min(numbers_per_process * (i + 1), N);
        counts[i] = (e - s);
        displ[i] = s;
    }

    int len = counts[rank];
    int *array_part = new int[len];

    MPI_Scatterv(array, counts, displ, MPI_INT, array_part, len, MPI_INT, root, MPI_COMM_WORLD);

    int max_part = -1;
    auto start_time = std::chrono::system_clock::now();

    for (int i = 0; i < len; ++i)
    {
        max_part = std::max(max_part, array_part[i]);
    }

    // print_array(array_part, len);
    // std::cout << max_part << ' ' << len << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::system_clock::now();

    MPI_Reduce(&max_part, &max, 1, MPI_INT, MPI_MAX, root, MPI_COMM_WORLD);

    if (rank == root)
    {
        std::cout << "Parallel max: " << duration(start_time, end_time) << " sec." << std::endl;
        std::cout << max << std::endl;
    }

    delete array_part;
    delete array;

    MPI_Finalize();
}