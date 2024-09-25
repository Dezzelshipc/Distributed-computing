#include <stdio.h>
#include <cstdlib>
#include "mpi.h"
#define NTIMES 1000000
int main(int argc, char **argv)
{
   double time_start, time_finish, tick;
   int rank, i;
   int len;
   char *name;
   name = (char*)malloc(MPI_MAX_PROCESSOR_NAME*sizeof(char));
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Get_processor_name(name, &len);
   tick = MPI_Wtick();
   time_start = MPI_Wtime();
   for (i = 0; i<NTIMES; i++)
      time_finish = MPI_Wtime();
   printf ("processor %s, process %d: tick= %lf, time= %lf\n", name, rank, tick, (time_finish-time_start));
   MPI_Finalize();
}