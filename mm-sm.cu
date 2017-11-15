/**
 * 
 * Matrix Multiplication - CUDA for GPUs
 *
 * CS3210
 *
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

int size;
#define BLOCK_SIZE 8

typedef struct
{
	float ** element;
} matrix;


long long wall_clock_time()
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/**
 * Allocates memory for a matrix of size SIZE
 * The memory is allocated row-major order, i.e. 
 *  elements from the same row are allocated at contiguous 
 *  memory addresses.
 **/
void allocate_matrix(matrix* m)
{
	int i;
	cudaError_t rc;
	
	// allocate array for all the rows
	rc = cudaMallocManaged((void**)&(m->element), sizeof(float*) * size);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}
	
	// allocate an array for each row of the matrix
	for (i = 0; i < size; i++)
	{
		rc = cudaMallocManaged((void**)&(m->element[i]), sizeof(float) * size);
		if (rc != cudaSuccess)
		{
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
			exit(1);
		}
	}
}

/**
 * Free the memory allocated for a matrix.
 **/
void free_matrix(matrix* m) {
	int i;
	for (i = 0; i < size; i++)
		cudaFree(m->element[i]);
	cudaFree(m->element);
}

/**
 * Initializes the elements of the matrix with
 * random values between 0 and 9
 **/
void init_matrix(matrix m)
{
	int i, j;
	
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = rand() % 10;
		}
}

/**
 * Initializes the elements of the matrix with
 * element 0.
 **/
void init_matrix_zero(matrix m)
{
	int i, j;
	
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = 0.0;
		}
}


/**
 * Multiplies matrix @a with matrix @b storing
 * the result in matrix @result
 * 
 * The multiplication algorithm is the O(n^3) 
 * algorithm
 */
void mm(matrix a, matrix b, matrix result)
{
	int i, j, k;
	
	// Do the multiplication
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			for(k = 0; k < size; k++)
				result.element[i][j] += a.element[i][k] * b.element[k][j];
}

__device__ float getElement(matrix a, int row, int col)
{
	return a.element[row][col];
}

__device__ void setElement(matrix a, int row, int col, float value)
{
	a.element[row][col] = value;
}

__device__ void free_block(matrix* m) {
	int i;
	for (i = 0; i < BLOCK_SIZE; i++)
		free(m->element[i]);
	free(m->element);
}

__device__ void malloc_block(matrix* m) {
		
	// allocate array for all the rows
	m->element = (float**) malloc(sizeof(float*) * BLOCK_SIZE);
	
}

__device__ void print_block(matrix* a)
{
	int i,j;
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
		printf("printing block...\n");
		for (i = 0; i < BLOCK_SIZE; i++) {
			printf("row %d :", i);
			for(j = 0; j < BLOCK_SIZE; j++) 
				printf("%1.2f ", a->element[i][j]);
			printf("\n");
		}
	}
}

__device__ void print_sm(float a[BLOCK_SIZE][BLOCK_SIZE]) 
{
	int i,j;
	int block = 1;
	if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == block && blockIdx.y == block) {
		printf("printing sm...\n");
		for (i = 0; i < BLOCK_SIZE; i++) {
			printf("row %d :", i);
			for(j = 0; j < BLOCK_SIZE; j++) 
				printf("%1.2f ", a[i][j]);
			printf("\n");
		}
	}
}

// row = block row
__device__ matrix getSubMatrix(matrix a, int row, int col) 
{
	matrix sub;
	int i;
	
	//malloc_block(&sub);
	//sub.element = (float**) malloc(sizeof(float *) * BLOCK_SIZE);
	// assign sub matrix elements
	for (i = 0; i < BLOCK_SIZE; i++) {
		sub.element[i] = a.element[row * BLOCK_SIZE + i] + (col * BLOCK_SIZE);
	}

	print_block(&sub);

	return sub;
}


/**
 * Each kernel computes the result element (i,j).
 */
__global__ void mm_kernel(matrix a, matrix b, matrix result, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k,e;

	if (i >= size || j >= size)
		return;

	//matrix Csub = getSubMatrix(result, blockIdx.y, blockIdx.x);
	int Cvalue = 0;
	
	for(k = 0; k < size/BLOCK_SIZE; k++) {
	
		//matrix Asub = getSubMatrix(a, blockIdx.y, k);
		//matrix Bsub = getSubMatrix(b, k, blockIdx.x);
		
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[threadIdx.y][threadIdx.x] = a.element[blockIdx.y * blockDim.y + threadIdx.y][k * BLOCK_SIZE + threadIdx.x];
		Bs[threadIdx.y][threadIdx.x] = b.element[k * BLOCK_SIZE + threadIdx.y][blockIdx.x * blockDim.x + threadIdx.x];	
		
		__syncthreads();
	
		for (e = 0; e < BLOCK_SIZE; e++) {
			//Cvalue += a.element[i][k * BLOCK_SIZE + e] * b.element[k * BLOCK_SIZE  + e][j];
			Cvalue += As[threadIdx.y][e] * Bs[e][threadIdx.x];
		}
		
		__syncthreads();

		//free_block(&Asub);
		//free_block(&Bsub);	
	}
	result.element[j][i] = Cvalue;
	//setElement(Csub, threadIdx.y, threadIdx.x, Cvalue);
	//free_block(&Csub);
}

void print_matrix(matrix m)
{
	int i, j;
	
	for (i = 0; i < size; i++)
	{
		printf("row %4d: ", i);
		for (j = 0; j < size; j++)
			printf("%6.2f  ", m.element[i][j]);
		printf("\n");
	}
}



void work()
{
	matrix a, b, result1, result2;
	long long before, after;
	int correct, i, j, dim;
	cudaError_t rc;

	// Allocate memory for matrices
	allocate_matrix(&a);
	allocate_matrix(&b);
	allocate_matrix(&result1);
	allocate_matrix(&result2);	

	// Initialize matrix elements
	init_matrix(a);
	init_matrix(b);
	
	// Perform sequential matrix multiplication
	before = wall_clock_time();
	mm(a, b, result1);
	after = wall_clock_time();
        fprintf(stderr, "Matrix multiplication on CPU took %1.2f seconds\n", ((float)(after - before))/1000000000);

	// Perform CUDA matrix  multiplication
	dim3 block(BLOCK_SIZE, BLOCK_SIZE);			// a block of BLOCK_SIZE CUDA threads
	dim = (size % BLOCK_SIZE == 0) ? size / BLOCK_SIZE : size / BLOCK_SIZE + 1; 
	dim3 grid(dim, dim);	// a grid of CUDA thread blocks
	before = wall_clock_time();
	mm_kernel<<<grid, block>>>(a, b, result2, size);
	cudaDeviceSynchronize();
	after = wall_clock_time();
	fprintf(stderr, "Matrix multiplication on GPU took %1.2f seconds\n", ((float)(after - before))/1000000000);

	// was there any error?
        rc = cudaGetLastError();
        if (rc != cudaSuccess)
                printf("Last CUDA error %s\n", cudaGetErrorString(rc));

	// Compare the results
	correct = 1;
	for (i = 0; correct && i < size; i++)
		for (j = 0; j < size; j++)
			if (result1.element[i][j] != result2.element[i][j]) {
				correct = 0;
				break;
			}

	if (correct) {
		printf("The result matrices are identical!\n");
	
	} else {
		printf("Difference in result matrices at element (%d, %d)!\n", i, j);
		//print_matrix(result1);
		//print_matrix(result2);
	}

	free_matrix(&a);
	free_matrix(&b);
	free_matrix(&result1);
	free_matrix(&result2);
}


int main(int argc, char ** argv)
{
	srand(0); 

	printf("Usage: %s <size>\n", argv[0]);
    
	if (argc >= 2)
		size = atoi(argv[1]);
	else
		size = 1024;
		
	fprintf(stderr,"Sequential matrix multiplication of size %d\n", size);
    
	// Multiply the matrices
	work();

	return 0;
}
