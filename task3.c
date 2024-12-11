#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows, cols;      // Dimensions of the matrix
    int* matrix = NULL;  // Matrix to be multiplied
    double scalar;       // Scalar value
    int* local_matrix;   // Local portion of the matrix
    int local_elements;  // Number of elements in the local portion
    double local_sum = 0.0, global_sum = 0.0;

    if (rank == 0) {
        // Process 0 reads the matrix dimensions and scalar
        printf("Enter the number of rows and columns of the matrix: ");
        scanf("%d %d", &rows, &cols);

        // Allocate memory for the matrix
        matrix = (int*)malloc(rows * cols * sizeof(int));

        printf("Enter the elements of the matrix: \n");
        for (int i = 0; i < rows * cols; i++) {
            scanf("%d", &matrix[i]);
        }

        printf("Enter the scalar value: ");
        scanf("%lf", &scalar);
    }

    // Broadcast the matrix dimensions and scalar to all processes
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&scalar, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate the total number of elements and the local size for each process
    int total_elements = rows * cols;
    local_elements = total_elements / size;
    if (rank < total_elements % size) {
        local_elements++; // Some processes handle one extra element
    }

    // Allocate memory for the local portion
    local_matrix = (int*)malloc(local_elements * sizeof(int));

    // Scatter the matrix to all processes
    int* send_counts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        send_counts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));

        int offset = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = total_elements / size;
            if (i < total_elements % size) {
                send_counts[i]++;
            }
            displs[i] = offset;
            offset += send_counts[i];
        }
    }

    MPI_Scatterv(matrix, send_counts, displs, MPI_INT, local_matrix, local_elements, MPI_INT, 0, MPI_COMM_WORLD);

    // Multiply the local portion by the scalar
    for (int i = 0; i < local_elements; i++) {
        local_matrix[i] = local_matrix[i] * scalar;
        local_sum += local_matrix[i];
    }

    // Gather the results back to process 0
    MPI_Gatherv(local_matrix, local_elements, MPI_INT, matrix, send_counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Reduce to find the global sum
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("The resulting matrix is: \n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", matrix[i * cols + j]);
            }
            printf("\n");
        }

        double average = global_sum / total_elements;
        printf("The average value of the matrix elements is: %.2f\n", average);

        free(matrix);
        free(send_counts);
        free(displs);
    }

    free(local_matrix);

    MPI_Finalize();
    return 0;
}
