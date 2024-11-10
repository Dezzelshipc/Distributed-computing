#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include "mpi.h"

#define N (uint64_t)400'000'000

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

    uint64_t dots_per_process = N / total_processes;
    if (rank == root) {
        std::cout << N << " " << total_processes << " " << dots_per_process << ' ' << std::endl;
        std::cout << RAND_MAX << std::endl;
    }
    uint64_t circle_dots = 0;

    auto start_time = std::chrono::system_clock::now();

    uint64_t circle_dots_p = 0;
    srand(time(NULL) + rank);

    for (int i = 0; i < dots_per_process; ++i)
    {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        circle_dots_p += (x * x + y * y < 1);
    }

    MPI_Reduce(&circle_dots_p, &circle_dots, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, root, MPI_COMM_WORLD);

    auto end_time = std::chrono::system_clock::now();

    if (rank == root)
    {
        double pi = 4 * (double)circle_dots / N;
        std::cout << std::setprecision(20) << pi << " " << circle_dots << " " << N
                  << "\nTime: " << duration(start_time, end_time) << " sec.\n";
    }

    MPI_Finalize();
}