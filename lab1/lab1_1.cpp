#include <iostream>
#include "mpi.h"

#define NTIMES 100
int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int number;
    if (rank % 2 == 0)
    {
        int total_processes;
        MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

        number = rank * 2;
        MPI_Send(&number, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
        std::cout << "Process " << rank << " sent message " << number << " to process " << rank+1 << "\n";
    }
    else
    {
        MPI_Recv(&number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << rank << " recieved message " << number << " from process 0\n";
    }

    MPI_Finalize();
}