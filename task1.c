#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure there are at least 2 processes
    if (size < 2) {
        if (rank == 0) {
            printf("This program requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Define the array abc
    double abc[10];
    double last_two[2]; // Array to store the last two elements

    if (rank == 0) {
        // Initialize the array abc with some values
        for (int i = 0; i < 10; i++) {
            abc[i] = i * 1.1; // Example initialization
        }

        // Extract the last two elements
        last_two[0] = abc[8];
        last_two[1] = abc[9];

        // Send the last two elements to all other processes
        for (int i = 1; i < size; i++) {
            MPI_Send(last_two, 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }

    } else {
        // Receive the last two elements on other processes
        MPI_Recv(last_two, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Store them in the same location as process 0
        double abc_received[10];
        abc_received[8] = last_two[0];
        abc_received[9] = last_two[1];

        // Print the received elements
        printf("Process %d received last two elements: %.1f, %.1f\n", rank, abc_received[8], abc_received[9]);
    }

    MPI_Finalize();
    return 0;
}
