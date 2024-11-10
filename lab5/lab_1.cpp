#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include "mpi.h"

#define N (uint64_t)500'000'000

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

    uint64_t total_dots = N * (uint64_t)total_processes;
    if (rank == root)
        std::cout << N << " " << total_processes << " " << N * total_processes << ' ' << std::endl;
    uint64_t circle_dots = 0;

    auto start_time = std::chrono::system_clock::now();

    uint64_t circle_dots_p = 0;

    //std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937_64 gen; // Standard mersenne_twister_engine seeded with rd()
    // std::uniform_real_distribution<> dis(0.0, 1.0);
    uint64_t max_ull = uint64_t(-1);
    std::uniform_int_distribution<uint64_t> dis(0, max_ull);

    for (int i = 0; i < N; ++i)
    {
        // double x = dis(gen);
        // double y = dis(gen);
        double x = (double)dis(gen)/ max_ull ;
        double y = (double)dis(gen)/ max_ull;
        circle_dots_p += (x * x + y * y < 1);
    }

    MPI_Reduce(&circle_dots_p, &circle_dots, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, root, MPI_COMM_WORLD);

    auto end_time = std::chrono::system_clock::now();

    if (rank == root)
    {
        double pi = 4 * (double)circle_dots / total_dots;
        std::cout <<std::setprecision(20) << pi << " " << circle_dots << " " << total_dots
                  << "\nTime: " << duration(start_time, end_time) << " sec.\n";
    }

    MPI_Finalize();
}