#include <iostream>
#include <vector>
#include <mpi.h>
#include <ctime>
#include <random>

void quick_sort(std::vector<int>& array, int left, int right) {
    int i, j, pivot, temp;

    if (left < right) {
        pivot = left;
        i = left;
        j = right;

        while (i < j) {
            while (array[i] <= array[pivot] && i <= right)
                i++;
            while (array[j] > array[pivot])
                j--;
            if (i < j) {
                temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        temp = array[pivot];
        array[pivot] = array[j];
        array[j] = temp;

        quick_sort(array, left, j - 1);
        quick_sort(array, j + 1, right);
    }
}

int main(int argc, char **argv) {
    int rank, size, a = 100; // Fixed array size
    std::vector<int> array, chunk;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        array.resize(a);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 1000); // Generate random numbers between 1 and 1000

        for (int i = 0; i < a; ++i)
            array[i] = dis(gen); // Assign random numbers to the array

        quick_sort(array, 0, a - 1);
    }

    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);

    chunk.resize(a / size);
    MPI_Scatter(array.data(), a / size, MPI_INT, chunk.data(), a / size, MPI_INT, 0, MPI_COMM_WORLD);

    clock_t start_time = clock();
    quick_sort(chunk, 0, a / size - 1);
    clock_t end_time = clock();

    double duration = (double)(end_time - start_time) / ((double)CLOCKS_PER_SEC / 1000000); // cast to double before computing duration
    if (rank == 0) {
        std::cout << "Execution time is " << duration << " microseconds\n";
    }

    for (int order = 1; order < size; order *= 2) {
        if (rank % (2 * order) != 0) {
            MPI_Send(chunk.data(), a / size, MPI_INT, rank - order, 0, MPI_COMM_WORLD);
            break;
        }
        int recv_size = (rank + order) < size ? a / size : a - (rank + order) * (a / size);

        std::vector<int> other(recv_size);
        MPI_Recv(other.data(), recv_size, MPI_INT, rank + order, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<int> temp(a / size + recv_size);
        int i = 0, j = 0, k = 0;
        while (i < a / size && j < recv_size) {
            if (chunk[i] < other[j])
                temp[k++] = chunk[i++];
            else
                temp[k++] = other[j++];
        }
        while (i < a / size)
            temp[k++] = chunk[i++];
        while (j < recv_size)
            temp[k++] = other[j++];

        chunk = temp;
    }

    if (rank == 0) {
        array.resize(a);
    }
    MPI_Gather(chunk.data(), a / size, MPI_INT, array.data(), a / size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Sorted array: ";
        for (int i = 0; i < a; ++i) {
            std::cout << array[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}
