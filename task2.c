#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n;               // Size of the vector
    int* vector = NULL;  // Vector to be multiplied
    double scalar;       // Scalar value
    int* local_vector;   // Local portion of the vector
    int local_size;      // Size of the local portion
    double local_sum = 0.0, global_sum = 0.0;

    if (rank == 0) {
        // Process 0 reads the vector size and scalar
        printf("Enter the size of the vector: ");
        scanf("%d", &n);

        // Allocate memory for the vector
        vector = (int*)malloc(n * sizeof(int));

        printf("Enter %d elements of the vector: ", n);
        for (int i = 0; i < n; i++) {
            scanf("%d", &vector[i]);
        }

        printf("Enter the scalar value: ");
        scanf("%lf", &scalar);
    }

    // Broadcast the vector size and scalar to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate the local size for each process
    local_size = n / size;
    if (rank < n % size) {
        local_size++; // Some processes handle one extra element
    }

    // Allocate memory for the local portion
    local_vector = (int*)malloc(local_size * sizeof(int));

    // Scatter the vector to all processes
    int* send_counts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        send_counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = n / size;
            if (i < n % size) {
                send_counts[i]++;
            }
            displs[i] = offset;
            offset += send_counts[i];
        }
    }

    MPI_Scatterv(vector, send_counts, displs, MPI_INT, local_vector, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply the local portion by the scalar
    for (int i = 0; i < local_size; i++) {
        local_vector[i] = local_vector[i] * scalar;
        local_sum += local_vector[i];
    }

    // Gather the results back to process 0
    MPI_Gatherv(local_vector, local_size, MPI_INT, vector, send_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Reduce to find the global sum
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The resulting vector is: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", vector[i]);
        }
        printf("\n");

        double average = global_sum / n;
        printf("The average value of the vector elements is: %.2f\n", average);

        free(vector);
        free(send_counts);
        free(displs);
    }

    free(local_vector);

    MPI_Finalize();
    return 0;
}
