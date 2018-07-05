#include <cmath>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/sigmoid_learn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out, Dtype rSlope, Dtype bias) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 0.5 * tanh(0.5 * (rSlope * (in[index] - bias)) ) + 0.5;
  }
}

template <typename Dtype>
void SigmoidLearnLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data_1 = top[0]->mutable_gpu_data();
  Dtype* top_data_2 = top[1]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* rSlope = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data_1, rSlope[0], bias[0]);
  CUDA_POST_KERNEL_CHECK;
  
  // test
  //LOG(INFO) << "0000000000000000000" << "\n";
  const Dtype* top_data_cpu = top[0]->cpu_data();
  const Dtype* bottom_data_cpu = bottom[0]->cpu_data();
  for (int i = 0; i < 10; i++){
  //LOG(INFO) << "forward top data " << top_data_cpu[i] << "\n";
  //LOG(INFO) << "forward bottom data " << bottom_data_cpu[i] << "\n";
  }
  //LOG(INFO) << "0000000000000000000" << "\n";

  Blob<Dtype>* ones = new Blob<Dtype>();
  ones->ReshapeLike(*top[0]);
  caffe_gpu_set(int(top[0]->count()), Dtype(1), ones->mutable_gpu_data());
  caffe_gpu_sub(int(top[0]->count()), ones->gpu_data(), top[0]->gpu_data(), top[1]->mutable_gpu_data());
  
  //LOG(INFO) << "rSlope: " << rSlope[0] << ", bias: " << bias[0] << ", top0: " << top[0]->cpu_data()[0] << ", top1: " << top[1]->cpu_data()[0] << "\n";
  ////LOG(INFO) << "----------------" << "\n";
  ////LOG(INFO) << "rSlope		" << rSlope[0] << "\n";
  ////LOG(INFO) << "bias		" << bias[0] << "\n";
  ////LOG(INFO) << "bottom		" << bottom[0]->cpu_data()[0] << "\n"; 
  ////LOG(INFO) << "top0		" << top[0]->cpu_data()[0] << "\n"; 
  ////LOG(INFO) << "top1		" << top[1]->cpu_data()[0] << "\n"; 
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void SigmoidBackward(const int n, const Dtype* out_diff_1, const Dtype* out_diff_2,
    const Dtype* out_data, Dtype* in_diff, Dtype rSlope) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    in_diff[index] = rSlope * (out_diff_1[index] - out_diff_2[index]) * sigmoid_x * (1 - sigmoid_x);
  }
}

template <typename Dtype>
__global__ void SigmoidParamBackward(const int n, const Dtype* out_diff_1, const Dtype* out_diff_2,
    const Dtype* out_data, const Dtype* in_data,
    Dtype* bias_diff, Dtype* rSlope_diff, Dtype rSlope, Dtype bias) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype sigmoid_x = out_data[index];
    rSlope_diff[index] = (in_data[index] - bias) * (out_diff_1[index] - out_diff_2[index]) * sigmoid_x * (1 - sigmoid_x);
    bias_diff[index] = (-1.) * rSlope * (out_diff_1[index] - out_diff_2[index]) * sigmoid_x * (1 - sigmoid_x);
    //bias_diff[0] += 1;
  }
}

template <typename Dtype>
void SigmoidLearnLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* rSlope = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff_1 = top[0]->gpu_diff();
    const Dtype* top_diff_2 = top[1]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    
    // test
    //LOG(INFO) << "!!!" << "\n";
    const Dtype* bias_diff_1 = this->blobs_[1]->cpu_diff();
    //LOG(INFO) << "bias shape " << this->blobs_[1]->shape_string() << "\n";
    //LOG(INFO) << "bias_diff " << bias_diff_1[0] << "\n";
    //LOG(INFO) << "!!!" << "\n";

    Dtype* rSlope_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    // change
    SigmoidParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff_1, top_diff_2, top_data, bottom_data, bias_diff_mat_.mutable_gpu_data(), rSlope_diff_mat_.mutable_gpu_data(), rSlope[0], bias[0]);  
    Dtype sum = 0;
    const Dtype* bias_diff_gpu = bias_diff_mat_.gpu_data();
    const Dtype* rSlope_diff_gpu = rSlope_diff_mat_.gpu_data();
    Blob<Dtype>* ones = new Blob<Dtype>();
    ones->ReshapeLike(bias_diff_mat_);
    caffe_gpu_set(int(ones->count()), Dtype(1), ones->mutable_gpu_data());
    caffe_gpu_dot(int(ones->count()), ones->gpu_data(), bias_diff_mat_.gpu_data(), &sum); 
    bias_diff[0] = sum;
    caffe_gpu_dot(int(ones->count()), ones->gpu_data(), rSlope_diff_mat_.gpu_data(), &sum); 
    rSlope_diff[0] = sum;
    
        

    // test
    //LOG(INFO) << "!!!" << "\n";
    const Dtype* bias_diff_2 = this->blobs_[1]->cpu_diff();
    //LOG(INFO) << "after backward bias_diff " << bias_diff_2[0] << "\n";
    //LOG(INFO) << "!!!" << "\n";
    // test 
    const Dtype* top_diff_1_1 = top[0]->cpu_diff();
    const Dtype* top_diff_2_2 = top[1]->cpu_diff();
    //LOG(INFO) << "================" << "\n";
    for (int i = 0; i <10 ; i++){
    //LOG(INFO) << "top_diff_1 " << top_diff_1_1[i] << "\n"; 
    //LOG(INFO) << "top_diff_2 " << top_diff_2_2[i] << "\n"; 
    }
    //LOG(INFO) << "================" << "\n";
    const Dtype* bias_diff_mat_cpu = bias_diff_mat_.cpu_data();
    for (int i = 0; i<10 ; i++){
    //LOG(INFO) << "bias_diff_mat '" << bias_diff_mat_cpu[i] << "\n";
    }
    //LOG(INFO) << "================" << "\n";
    const Dtype* top_data_cpu = top[0]->cpu_data();
    for (int i = 0; i<10; i++){
    //LOG(INFO) << "top_data " << top_data_cpu[i] << "\n";
    }
    //LOG(INFO) << "================" << "\n";
  }
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff_1 = top[0]->gpu_diff();
    const Dtype* top_diff_2 = top[1]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff_1, top_diff_2, top_data, bottom_diff, rSlope[0]);
    //const Dtype* bias_diff = bias_diff_mat_.cpu_data();
    //const Dtype* rSlope_diff = rSlope_diff_mat_.cpu_data();
    //for (int i = 0; i < int(bias_diff_mat_.count()); i++) {
    //  bias_ -= (bias_diff[i] * bias_lr_mult_);
    //  rSlope_ -= (rSlope_diff[i] * rSlope_lr_mult_);
    //}
    CUDA_POST_KERNEL_CHECK;
  }
    const Dtype* r_diff = this->blobs_[0]->cpu_diff();
    const Dtype* b_diff = this->blobs_[1]->cpu_diff();
    ////LOG(INFO) << "bias diff	"<< b_diff[0] << "\n";
    ////LOG(INFO) << "rSlope diff 	"<< r_diff[0] << "\n";
    ////LOG(INFO) << "bottom diff	"<< bottom[0]->cpu_diff()[0] << "\n";
    ////LOG(INFO) << "top0 diff	"<< top[0]->cpu_diff()[0]  << "\n";
    ////LOG(INFO) << "top1 diff	"<< top[1]->cpu_diff()[0]  << "\n";
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidLearnLayer);


}  // namespace caffe
