#include <iostream>
#include <chrono>
#include <omp.h>

const int size = 500;
int matrix1[size][size] = { 0 };
int matrix2[size][size] = { 0 };
int matrix_result[size][size] = { 0 };

void sequential_matrix_multiplication()
{
	for (int i = 0;i < size;i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix1[i][j] = rand() % 10;
			matrix2[i][j] = rand() % 10;
		}
	}
	//ijk
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				matrix_result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto mcsec = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "ijk: " << mcsec.count() << " microseconds\n";

	for (int i = 0;i < size;i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix_result[i][j] = 0;
		}
	}

	//kij
	auto start1 = std::chrono::high_resolution_clock::now();
	for (int k = 0; k < size; k++) {
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				matrix_result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
	auto end1 = std::chrono::high_resolution_clock::now();
	auto mcsec1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1);
	std::cout << "kij(optimal): " << mcsec1.count() << " microseconds\n";


	for (int i = 0;i < size;i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix_result[i][j] = 0;
		}
	}
}

void openMP_matrix_multiplication()
{

	for (int i = 0;i < size;i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix1[i][j] = rand() % 10;
			matrix2[i][j] = rand() % 10;
		}
	}

	auto start = std::chrono::high_resolution_clock::now();

	int i, j, k;

#pragma omp parallel for shared(matrix1,matrix2,matrix_result) private(i, j, k)
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				matrix_result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto mcsec = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
	std::cout << "openMP: " << mcsec.count() << " microseconds\n";


	for (int i = 0;i < size;i++)
	{
		for (int j = 0; j < size; j++)
		{
			matrix_result[i][j] = 0;
		}
	}
}


int main()
{
	sequential_matrix_multiplication();
	openMP_matrix_multiplication();
	return 0;
}
