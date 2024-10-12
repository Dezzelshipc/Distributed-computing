#include <iostream>
#include "mpi.h"

int main(int argc, char **argv)
{
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int number;
    if (rank == 0)
    {
        int total_processes;
        MPI_Comm_size(MPI_COMM_WORLD, &total_processes);

        for (int i = 1; i < total_processes; ++i)
        {
            number = i * 2;
            MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            std::cout << "Process 0 sent message " << number << " to process " << i << "\n";
        }
    }
    else
    {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Process " << rank << " recieved message " << number << " from process 0\n";
    }

    MPI_Finalize();
}