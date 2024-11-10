#include <iostream>
#include <chrono>
#include <iomanip>
#include "mpi.h"

#define N 1000
#define M 100

double duration(auto start_time, auto end_time)
{
    return std::chrono::duration<double>(end_time - start_time).count();
}

void print_mat(int *mat)
{
    if (N <= 4)
    {
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                std::cout << mat[i * N + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

bool compare_mat(int *mat1, int *mat2)
{
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (mat1[i * N + j] != mat2[i * N + j])
            {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char **argv)
{
    int rank;
    int total_processes;
    const int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

    int *mat1 = new int[N * N];
    int *mat2 = new int[N * N];
    int *mat3 = new int[N * N];
    int *mat4 = new int[N * N];

    if (rank == root)
    {
        srand(time(NULL));
        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                mat1[i * N + j] = rand() % M;
                mat2[i * N + j] = rand() % M;
                mat3[i * N + j] = 0;
                mat4[i * N + j] = 0;
            }
        }

        auto start_time = std::chrono::system_clock::now();

        for (int i = 0; i < N; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                int sum = 0;
                for (int k = 0; k < N; ++k)
                {
                    sum += mat1[i * N + k] * mat2[k * N + j];
                }
                mat3[i * N + j] = sum;
            }
        }

        auto end_time = std::chrono::system_clock::now();

        std::cout << "Non-parallel matrix mult: " << duration(start_time, end_time) << " sec." << std::endl;
    }

    MPI_Bcast(mat1, N * N, MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast(mat2, N * N, MPI_INT, root, MPI_COMM_WORLD);

    int counts[N];
    int displ[N];

    int rows_per_process = N / total_processes;
    if (N % total_processes != 0)
    {
        rows_per_process += 1;
    }
    int start_row = rows_per_process * rank;
    int end_row = std::min(rows_per_process * (rank + 1), N);

    for (int i = 0; i < N; ++i)
    {
        int s = rows_per_process * i;
        int e = std::min(rows_per_process * (i + 1), N);
        counts[i] = (e - s) * N;
        displ[i] = s * N;
    }

    int len = counts[rank];

    int *mat4_part = new int[len];

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::system_clock::now();

    for (int i = 0; i < end_row - start_row; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += mat1[(i + start_row) * N + k] * mat2[k * N + j];
            }
            mat4_part[i * N + j] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::system_clock::now();

    MPI_Gatherv(mat4_part, len, MPI_INT, mat4, counts, displ, MPI_INT, root, MPI_COMM_WORLD);

    delete mat4_part;

    if (rank == root)
    {
        std::cout << "Parallel matrix mult: " << duration(start_time, end_time) << " sec." << std::endl;

        std::cout << "Mat 1\n";
        print_mat(mat1);
        std::cout << "Mat 2\n";
        print_mat(mat2);
        std::cout << "Mat consecutive\n";
        print_mat(mat3);
        std::cout << "Mat parallel\n";
        print_mat(mat4);

        if (compare_mat(mat3, mat4))
        {
            std::cout << "Same mat\n";
        }
        else
        {
            std::cout << "Different mat\n";
        }
    }

    MPI_Finalize();
}