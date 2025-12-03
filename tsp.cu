#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

const int MAXN = 16;                 // Maximum number of cities
const int THREADS_PER_BLOCK = 256;   // Threads per block
const int BLOCKS = 64;               // Number of blocks

__managed__ long long factorial[MAXN+1];
__managed__ int block_best_cost[BLOCKS];
__managed__ long long block_best_perm[BLOCKS];

// Compute factorials (1,2,...,N)
void compute_factorial(int N) {
    factorial[0] = 1;
    for (int i=1; i<=N; i++) factorial[i] = factorial[i-1] * i;
}

// Generate random symmetric distance matrix
void generate_matrix(int* matrix, int N, int min_weight=1, int max_weight=10) {
    for (int i=0;i<N;i++){
        for (int j=i+1;j<N;j++){
            int w = min_weight + rand() % (max_weight - min_weight + 1);
            matrix[i*N+j] = matrix[j*N+i] = w;
        }
        matrix[i*N+i] = 0;
    }
}

// Find cost of a path
__device__ int path_cost(int* matrix, int* path, int N) {
    int cost = 0;
    for(int i=1;i<N;i++){
        cost += matrix[path[i-1]*N + path[i]];
    }
    cost += matrix[path[N-1]*N + path[0]]; // return to start
    return cost;
}

// Convert permutation index to permutation (factoradic / lexicographic)
__device__ void decode_perm(int* perm, int n, long long index) {
    index--; // 1-based to 0-based
    int pool[MAXN];
    long long fact[MAXN];
    fact[0] = 1;
    for(int i=1;i<n;i++) fact[i] = fact[i-1]*i;
    for(int i=0;i<n;i++) pool[i]=i;
    
    for(int i=0;i<n;i++){
        long long f = index / fact[n-1-i];
        index %= fact[n-1-i];
        perm[i] = pool[f];
        for(int j=f;j<n-1-i;j++) pool[j]=pool[j+1];
    }
}

// CUDA kernel
__global__ void tsp_kernel(int* d_matrix, int N, long long total_perms) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    long long per_thread = total_perms / (gridDim.x*blockDim.x);
    long long start_perm = tid * per_thread + 1;
    if(tid == gridDim.x*blockDim.x - 1) per_thread += total_perms % (gridDim.x*blockDim.x);

    int perm[MAXN];
    int best_cost = 1e9;
    long long best_index = start_perm;

    for(long long i=0;i<per_thread;i++){
        decode_perm(perm, N, start_perm+i);
        int cost = path_cost(d_matrix, perm, N);
        if(cost < best_cost){
            best_cost = cost;
            best_index = start_perm+i;
        }
    }

    __shared__ int shared_cost[THREADS_PER_BLOCK];
    __shared__ long long shared_index[THREADS_PER_BLOCK];
    int lane = threadIdx.x;
    shared_cost[lane] = best_cost;
    shared_index[lane] = best_index;
    __syncthreads();

    // Reduction to find block best
    for(int s=blockDim.x/2; s>0; s>>=1){
        if(lane < s){
            if(shared_cost[lane+s] < shared_cost[lane]){
                shared_cost[lane] = shared_cost[lane+s];
                shared_index[lane] = shared_index[lane+s];
            }
        }
        __syncthreads();
    }

    if(lane==0){
        block_best_cost[blockIdx.x] = shared_cost[0];
        block_best_perm[blockIdx.x] = shared_index[0];
    }
}

void print_matrix(int* matrix, int N) {
    cout << "Distance Matrix (" << N << "x" << N << "):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i * N + j] << "\t";
        }
        cout << "\n";
    }
    cout << endl;
}

int main(int argc, char** argv){
    if(argc < 2){
        cout << "Usage: ./tsp N\n"; return 1;
    }
    int N = stoi(argv[1]);
    if(N>MAXN){ cout << "N too large for brute force!\n"; return 1; }

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    compute_factorial(N);
    
    long long total_perms = factorial[N];

    int* h_matrix = new int[N*N];
    generate_matrix(h_matrix, N);
	print_matrix(h_matrix, N);

    int* d_matrix;
    cudaMalloc(&d_matrix, sizeof(int)*N*N);
    cudaEventRecord(start);
    cudaMemcpy(d_matrix, h_matrix, sizeof(int)*N*N, cudaMemcpyHostToDevice);

    // Initialize block best costs
    for(int i=0;i<BLOCKS;i++) block_best_cost[i]=1e9;

    tsp_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_matrix, N, total_perms);
    cudaDeviceSynchronize();

    // Find final best among blocks
    int best_cost = 1e9;
    long long best_perm = 1;
    for(int i=0;i<BLOCKS;i++){
        if(block_best_cost[i] < best_cost){
            best_cost = block_best_cost[i];
            best_perm = block_best_perm[i];
        }
    }

	cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%f\n", milliseconds*0.001);

    int final_perm[MAXN];
    // Decode on host
    auto decode_host = [&](int* perm, int n, long long idx){
        idx--;
        int pool[MAXN];
        long long fact[MAXN];
        fact[0]=1;
        for(int i=1;i<n;i++) fact[i]=fact[i-1]*i;
        for(int i=0;i<n;i++) pool[i]=i;
        for(int i=0;i<n;i++){
            long long f = idx / fact[n-1-i];
            idx %= fact[n-1-i];
            perm[i]=pool[f];
            for(int j=f;j<n-1-i;j++) pool[j]=pool[j+1];
        }
    };
    decode_host(final_perm, N, best_perm);

    cout << "Minimum cost: " << best_cost << "\n";
    cout << "Path: ";
    for(int i=0;i<N + 1;i++) cout << final_perm[i] << " ";
    cout << final_perm[0] << "\n"; // return to start

    delete[] h_matrix;
    cudaFree(d_matrix);
}
