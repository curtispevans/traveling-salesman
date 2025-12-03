#include <bits/stdc++.h>
using namespace std;

const int THREADS_PER_BLOCK = 1024;
const int BLOCKS = 50;

const int MAXN = 16;
const int INF = 1e9;
const int MIN_EDGE_WEIGHT = 1;
const int MAX_EDGE_WEIGHT = 10;

long long factorial_arr[MAXN+1];

int block_optimal_values[BLOCKS];
int block_optimal_permutation[BLOCKS];

//////////////////////////////////////////////////////////////
// Random helpers
//////////////////////////////////////////////////////////////

int random_int(int l, int r) {
    return l + rand() % (r - l + 1);
}

void precompute_factorial() {
    factorial_arr[0] = 1;
    for(int i=1;i<=MAXN;i++)
        factorial_arr[i] = factorial_arr[i-1] * i;
}

void assign_edge_weights(int* matrix, int N) {
    for(int i=0;i<N;i++) {
        for(int j=i+1;j<N;j++) {
            int w = random_int(MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT);
            matrix[i*N+j] = w;
            matrix[j*N+i] = w;
        }
        matrix[i*N+i] = 0;
    }
}

void print_matrix(int* matrix, int N) {
    for (int i=0;i<N;i++) {
        for(int j=0;j<N;j++)
            cout << matrix[i*N+j] << " ";
        cout << "\n";
    }
}

//////////////////////////////////////////////////////////////
// Permutation logic (CPU-only versions)
//////////////////////////////////////////////////////////////

bool nxt_permutation_cpu(int *arr, int n) {
    int fi = -1;
    for(int i=n-2;i>=0;i--) {
        if(arr[i] < arr[i+1]) {
            fi = i;
            break;
        }
    }
    if(fi < 0) return false;

    int ng = INT_MAX, ngi = -1;
    for(int i=fi+1;i<n;i++) {
        if(arr[i] > arr[fi] && arr[i] < ng) {
            ng = arr[i];
            ngi = i;
        }
    }

    swap(arr[fi], arr[ngi]);
    reverse(arr + fi + 1, arr + n);
    return true;
}

long long fact(int n) {
    long long r = 1;
    for(int i=1;i<=n;i++) r *= i;
    return r;
}

// arr must be sorted initially
bool nth_permutation_cpu(int *arr, int arrsize, long long n) {
    if(n > fact(arrsize)) return false;

    bool taken[MAXN] = {false};
    int *ans = new int[arrsize];

    for(int i=0;i<arrsize;i++) {
        long long block = fact(arrsize - 1 - i);
        int idx = (n - 1) / block;
        n -= idx * block;

        int c = -1;
        for(int j=0;j<arrsize;j++) {
            if(!taken[j]) {
                if(++c == idx) {
                    ans[i] = arr[j];
                    taken[j] = true;
                    break;
                }
            }
        }
    }

    for(int i=0;i<arrsize;i++) arr[i] = ans[i];
    delete[] ans;
    return true;
}

int find_path_cost(int* matrix, int* arr, int arrsize, int N) {
    int cost = 0;
    for(int i=1;i<arrsize;i++)
        cost += matrix[arr[i]*N + arr[i-1]];
    return cost;
}

//////////////////////////////////////////////////////////////
// CPU simulation of CUDA kernel
//////////////////////////////////////////////////////////////

void tsp_cpu(int* matrix, int* path, int N) {
    long long total_perm = factorial_arr[N-1];
    long long per_thread = total_perm / (BLOCKS * THREADS_PER_BLOCK);

    // initialize block best values
    for(int i=0;i<BLOCKS;i++) {
        block_optimal_values[i] = INF;
        block_optimal_permutation[i] = 1;
    }

    // Simulate each CUDA thread using a normal loop
    for(int thread = 0; thread < BLOCKS * THREADS_PER_BLOCK; thread++) {

        int block = thread / THREADS_PER_BLOCK;

        long long start_perm = thread * per_thread + 1;
        long long iter_cnt = per_thread;

        if(thread == BLOCKS * THREADS_PER_BLOCK - 1)
            iter_cnt += total_perm % (BLOCKS * THREADS_PER_BLOCK);

        int arr[MAXN];
        for(int i=1;i<N;i++) arr[i-1] = path[i];

        nth_permutation_cpu(arr, N-1, start_perm);

        int best_val = INF;
        long long best_perm = start_perm;

        long long iter = 0;
        do {
            int full_path[MAXN+1];
            full_path[0] = 0;
            for(int i=1;i<N;i++)
                full_path[i] = arr[i-1];
            full_path[N] = 0;

            int val = find_path_cost(matrix, full_path, N+1, N);

            if(val < best_val) {
                best_val = val;
                best_perm = start_perm + iter;
            }

            iter++;

        } while(iter < iter_cnt && nxt_permutation_cpu(arr, N-1));

        if(best_val < block_optimal_values[block]) {
            block_optimal_values[block] = best_val;
            block_optimal_permutation[block] = best_perm;
        }
    }
}

//////////////////////////////////////////////////////////////
// MAIN (No CUDA anywhere)
//////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    if(argc < 2) {
        cout << "Usage: ./tsp N\n";
        return 0;
    }

    int N = stoi(argv[1]);

    precompute_factorial();

    int* matrix = new int[N*N];

    int path[MAXN];
    path[0] = 0;
    path[N] = 0;
    for(int i=1;i<N;i++) path[i] = i;

    assign_edge_weights(matrix, N);

    print_matrix(matrix, N);

    auto start = chrono::high_resolution_clock::now();

    // CPU simulation of CUDA
    tsp_cpu(matrix, path, N);

    // Collect global best
    int best_cost = INF;
    long long best_perm = 1;

    for(int i=0;i<BLOCKS;i++) {
        if(block_optimal_values[i] < best_cost) {
            best_cost = block_optimal_values[i];
            best_perm = block_optimal_permutation[i];
        }
    }

    // reconstruct permutation
    int arr[MAXN];
    for(int i=1;i<N;i++) arr[i-1] = path[i];
    nth_permutation_cpu(arr, N-1, best_perm);
    for(int i=1;i<N;i++) path[i] = arr[i-1];

    auto stop = chrono::high_resolution_clock::now();
    double seconds = chrono::duration<double>(stop - start).count();

    cout << "Time: " << seconds << " s\n";

    cout << "Minimum Cost Path: ";
    for(int i=0;i<N;i++) cout << path[i] << " ";
    cout << path[0] << "\n";

    int cost = 0;
    for(int i=1;i<=N;i++)
        cost += matrix[path[i]*N + path[i-1]];

    cout << "Path cost: " << cost << "\n";

    delete[] matrix;
}
