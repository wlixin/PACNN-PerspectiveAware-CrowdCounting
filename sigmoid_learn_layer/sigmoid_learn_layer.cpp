#include <cmath>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/sigmoid_learn_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x, Dtype rSlope_, Dtype bias_) {
  Dtype tmp = rSlope_ * (x - bias_);
  return 0.5 * tanh(0.5 * tmp) + 0.5;
}

template <typename Dtype>
  void SigmoidLearnLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //SigmoidLearnParameter sigmoid_learn_param = this->layer_param_.sigmoid_learn_param();
    SigmoidLearnParameter sigmoid_learn_param = this->layer_param().sigmoid_learn_param();
    //bias_lr_mult_ = sigmoid_learn_param.bias_lr_mult();
    //rSlope_lr_mult_ = sigmoid_learn_param.rslope_lr_mult();

    //bias_ = 30;
    //rSlope_ = 0.5;
    // blobs_[0] holds the rSlope 
    // blobs_[1] holds the bias
    if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping paramter initialization";
    } else {
      this->blobs_.resize(2);
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
      this->blobs_[1].reset(new Blob<Dtype>(vector<int>(0)));
      shared_ptr<Filler<Dtype> > rSlope_filler;
      shared_ptr<Filler<Dtype> > bias_filler;
      if (sigmoid_learn_param.has_rslope_filler() && sigmoid_learn_param.has_bias_filler()) {
        rSlope_filler.reset(GetFiller<Dtype>(sigmoid_learn_param.rslope_filler()));
        bias_filler.reset(GetFiller<Dtype>(sigmoid_learn_param.bias_filler()));
      } else {
        FillerParameter rSlope_filler_param;
        rSlope_filler_param.set_type("constant");
        rSlope_filler_param.set_value(0.5);
        rSlope_filler.reset(GetFiller<Dtype>(rSlope_filler_param));

        FillerParameter bias_filler_param;
        bias_filler_param.set_type("constant");
        bias_filler_param.set_value(30);
        bias_filler.reset(GetFiller<Dtype>(bias_filler_param));
      }
      rSlope_filler->Fill(this->blobs_[0].get());
      bias_filler->Fill(this->blobs_[1].get());

      this->param_propagate_down_.resize(this->blobs_.size(), true);
    }
  }

  template <typename Dtype>
  void SigmoidLearnLayer<Dtype>::Reshape(const vector <Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    top[1]->ReshapeLike(*bottom[0]);
    
    bias_diff_mat_.ReshapeLike(*top[0]);
    rSlope_diff_mat_.ReshapeLike(*top[0]);
  }

template <typename Dtype>
void SigmoidLearnLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data_1 = top[0]->mutable_cpu_data();
  Dtype* top_data_2 = top[1]->mutable_cpu_data();
  const int count = bottom[0]->count();
 ////////////////
  const Dtype* rSlope = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();
  for (int i = 0; i < count; ++i) {
    top_data_1[i] = sigmoid(bottom_data[i], rSlope[0], bias[0]);
    top_data_2[i] = 1 - top_data_1[i];
  }
}

template <typename Dtype>
void SigmoidLearnLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* rSlope = this->blobs_[0]->cpu_data();
  const Dtype* bias = this->blobs_[1]->cpu_data();
  if (this->param_propagate_down_[0]) {
    Dtype* rSlope_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    //Dtype rSlope_diff = 0;
    //Dtype bias_diff = 0;
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff_1 = top[0]->cpu_diff();
    const Dtype* top_diff_2 = top[1]->cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      // bottom_diff[i] = rSlope_ * top_diff[i] * sigmoid_x * (1. - sigmoid_x);
      // rSlope_diff += (bottom_data[i] - bias_) * top_diff[i] * sigmoid_x * (1. - sigmoid_x);
      // bias_diff += (-1.) * rSlope_ * top_diff[i] * sigmoid_x * (1. - sigmoid_x);
      rSlope_diff[0] += (bottom_data[i] - bias[0]) * (top_diff_1[i] - top_diff_2[i]) * sigmoid_x * (1. - sigmoid_x);
      bias_diff[0] += (-1.) * rSlope[0] * (top_diff_1[i] - top_diff_2[i]) * sigmoid_x * (1. - sigmoid_x);
    }
    //rSlope_ -= (rSlope_diff * rSlope_lr_mult_);
    //bias_ -= (bias_diff * bias_lr_mult_);
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff_1 = top[0]->cpu_diff();
    const Dtype* top_diff_2 = top[1]->cpu_diff();
    const Dtype* top_data = top[0]->cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = rSlope[0] * (top_diff_1[i] - top_diff_2[i]) * sigmoid_x * (1. - sigmoid_x);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SigmoidLearnLayer);
#endif

INSTANTIATE_CLASS(SigmoidLearnLayer);
REGISTER_LAYER_CLASS(SigmoidLearn);

}  // namespace caffe
