extern "C" {
#include <math.h>
#include <stdio.h>

struct DeviceMatrix {
    float* data;
    unsigned int row_len;
    unsigned int data_len;
};

// LINEAR ALGEBRA OPERATIONS ---------------
__global__ void scale(float* mat_a, float* mat_c, float s, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    
    mat_c[idx] = mat_a[idx] * s;
}

__global__ void offset(float* mat_a, float* mat_c, float o, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    
    mat_c[idx] = mat_a[idx] + o;
}

__global__ void dot(
    float* dot, float* mat_a, float* mat_b, unsigned int row_len, unsigned int n
    ) {

    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    if (id_x >= row_len) return;

    float sum = 0.0f;
    for (int i=0; i<n/row_len; i++){
        sum += mat_a[id_x + i*row_len] * mat_b[id_x + i*row_len];
    }
    dot[id_x] = sum;
}

__global__ void internal_mean(
    float* mat_a, float* vec_c, unsigned int row_len, unsigned int n
    ) {

    unsigned int id_y = threadIdx.x + blockDim.x * blockIdx.x;

    int col_len = n/row_len;
    if (id_y >= col_len) return;

    float sum = 0.0f;

    for (int i=0; i<row_len; i++){
        sum += mat_a[i + id_y*row_len];
    }
    vec_c[id_y] = sum / row_len;
}

__global__ void component_add(const float* mat_a, const float* mat_b, float* mat_c, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    mat_c[idx] = mat_a[idx] + mat_b[idx];
}

__global__ void component_mul(const float* mat_a, const float* mat_b, float* mat_c, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    mat_c[idx] = mat_a[idx] * mat_b[idx];
}

__global__ void multiply(
    const float* mat_a, const float* mat_b, float* mat_c,
    unsigned int a_row_len, unsigned int b_row_len, unsigned int a_col_len){
    
    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int id_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (id_x >= b_row_len || id_y >= a_col_len) return;
    
    
    float dot = 0.0f;
    for (int i=0; i<a_row_len; i++){
        dot += mat_a[i + id_y*a_row_len] * mat_b[id_x + i*b_row_len];
    }

    mat_c[id_x + id_y*b_row_len] = dot;
}

__global__ void transpose(const float* mat_a, float* mat_c, unsigned int row_len, unsigned int n){
    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int id_y = threadIdx.y + blockDim.y * blockIdx.y;
    if (id_x >= row_len || id_y >= n/row_len) return;

    unsigned int c_row_len = n/row_len;

    mat_c[id_y + id_x * c_row_len] = mat_a[id_x + id_y*row_len];
}

// ACTIVATION FUNCTIONS --------------------

__global__ void ReLU(float* mat_a, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    if (mat_a[idx] < 0){
        mat_a[idx] = 0;
    }
}

__global__ void ReLU_inf(float* vec_a, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    if (vec_a[idx] < 0){
        vec_a[idx] = 0;
    }
    else{
        vec_a[idx] = 1;
    }
}

__global__ void Sigmoid(float* mat_a, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    mat_a[idx] = 1 / (1 + exp(-mat_a[idx]));
}

__global__ void Sigmoid_inf(float* vec_a, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    float sig = 1 / (1 + exp(-vec_a[idx]));
    vec_a[idx] = sig * (1-sig);
}

__global__ void Tanh(float* mat_a, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    mat_a[idx] = tanh(mat_a[idx]);
}

__global__ void Tanh_inf(float* vec_a, unsigned int n) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= n) return;
    float t = tanh(vec_a[idx]);
    vec_a[idx] = 1-(t*t);
}

__global__ void SoftMax(float* mat_a, unsigned int a_row_len, unsigned int n) {
    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    if (id_x >= a_row_len) return;

    float element;
    float tot;
    for (int i=0; i<n/a_row_len; i++){
        element = exp(mat_a[id_x + i*a_row_len]);
        mat_a[id_x + i*a_row_len] = element;
        tot += element;
    }

    for (int i=0; i<n/a_row_len; i++){
        mat_a[id_x + i*a_row_len] /= tot;
    }
}

// LOSS FUNCTIONS --------------------------

__global__ void Mean_Squared_Error(
    float* vec_c, float* mat_output, float* mat_expected, unsigned int row_len, unsigned int n
    ) {

    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;

    if (id_x >= row_len) return;
    
    float sum = 0.0f;
    for (int i=0; i<n/row_len; i++){
        float element = (mat_output[id_x + i*row_len] - mat_expected[id_x + i*row_len]);
        sum += element*element;
    }
    vec_c[id_x] = sum;
}

__global__ void Cross_Entropy(
    float* vec_c, float* mat_output, float* mat_expected, unsigned int row_len, unsigned int n
    ) {

    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;

    if (id_x >= row_len) return;
    float sum = 0.0f;
    for (int i=0; i<n/row_len; i++){
        if (mat_expected[id_x + i*row_len] != 0){
            float element = mat_expected[id_x + i*row_len] * log(mat_output[id_x + i*row_len]);
            sum += element;
        }
    }
    vec_c[id_x] = -sum;
}

// STATISTIC FUNCTIONS ---------------------

__global__ void Accuracy(
    bool* vec_c, float* mat_output, float* mat_expected, unsigned int row_len, unsigned int n
    ) {

    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;

    if (id_x >= row_len) return;
    float max = -INFINITY;
    int index_o = 0;

    for (int i=0; i<n/row_len; i++){
        float element = mat_output[id_x + i*row_len];
        if (element > max){
            max = element;
            index_o = i;
        }
    }

    if (mat_expected[id_x + index_o*row_len] > 0){
        vec_c[id_x] = true;
    }
    else{
        vec_c[id_x] = false;
    }
}

__global__ void Add_Bias(float* mat_a, float* bias, unsigned int row_len, unsigned int n) {
    unsigned int id_x = threadIdx.x + blockDim.x * blockIdx.x;
    if (id_x >= row_len) return;

    for (int i=0; i<n/row_len; i++){
        mat_a[id_x + i*row_len] += bias[i];
    }
}

}