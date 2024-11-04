#include <vector>
#include <thread>
#include <future>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;


// BasicUnitNMSVector class definition using std::vector
class BasicUnitNMSVector {
public:
    BasicUnitNMSVector(const std::vector<std::vector<std::vector<float>>>& logits, const std::vector<int>& expert_kernel_sizes, const std::vector<std::vector<int>>& attention_mask)
    : logits(logits), expert_kernel_sizes(expert_kernel_sizes), attention_mask(attention_mask) {
        batch_size = logits.size();
        seq_len = logits[0].size();
        num_experts = logits[0][0].size();
        basic_unit_mask_center.resize(batch_size, std::vector<int>(seq_len, 0));
        basic_unit_mask_all.resize(batch_size, std::vector<int>(seq_len, 0));
        lengths.resize(batch_size, 0);
        for (size_t i = 0; i < batch_size; ++i) {
            lengths[i] = std::accumulate(attention_mask[i].begin(), attention_mask[i].end(), 0);
        }
    }

    void basic_unit_nms_sort_based(int i, int left, int right) {
        // Do non-maximum suppression based on the logits with threshold = 0.0 (Non overlapping basic_units)
        // only focues within the range of left and right
        std::vector<int> sorted_indices( (right - left) * num_experts);
        std::iota( sorted_indices.begin(), sorted_indices.end(), 0 );
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return logits[i][a / num_experts + left][a % num_experts] > logits[i][b / num_experts + left][b % num_experts];
        });
        for (auto index : sorted_indices) {
            int position = index / num_experts + left;
            if (basic_unit_mask_all[i][position] != 0) {
                continue;
            }
            int expert = index % num_experts;
            int kernel_size = expert_kernel_sizes[expert];
            int left_boundary =  position - kernel_size / 2;
            int right_boundary =  position + kernel_size / 2 + kernel_size % 2;
            if (left_boundary < left || right_boundary > right) {
                continue;
            }
            // Check whether overlap with previous basic_units
            auto summation = std::accumulate(basic_unit_mask_all[i].begin() + left_boundary, basic_unit_mask_all[i].begin() + right_boundary, 0);
            if (summation == 0) {
                basic_unit_mask_center[i][position] = expert+1;
                std::fill(basic_unit_mask_all[i].begin() + left_boundary,basic_unit_mask_all[i].begin() + right_boundary, expert+1);
            }
        }
    }

    std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> batched_basic_unit_nms_sorted() {
        // Parallelize the non-maximum suppression across samples in the batch
        std::vector<std::future<void>> futures;
        for (int i = 0; i < batch_size; ++i) {
            futures.push_back(std::async(std::launch::async, &BasicUnitNMSVector::basic_unit_nms_sort_based, this, i, 1, lengths[i] - 1)); // Discard the [CLS] and [SEP] tokens
        }
        for (auto& future : futures) {
            future.get(); // Synchronize all threads
        }
        return {basic_unit_mask_center, basic_unit_mask_all};
    }

private:
    std::vector<std::vector<std::vector<float>>> logits;
    std::vector<int> expert_kernel_sizes;
    std::vector<std::vector<int>> basic_unit_mask, basic_unit_mask_center, basic_unit_mask_all, attention_mask;
    std::vector<int> lengths;
    size_t batch_size, seq_len, num_experts;
};

std::pair<py::array_t<int>,py::array_t<int>> basic_unit_nms(py::array_t<float> logits_arr, py::array_t<int> expertKernelSizes_arr, py::array_t<int> attentionMask_arr)
{
        py::buffer_info logits_buf = logits_arr.request();
        py::buffer_info expert_sizes_buf = expertKernelSizes_arr.request();
        py::buffer_info mask_buf = attentionMask_arr.request();

        if (logits_buf.ndim != 3)
            throw std::runtime_error("Expected a 3-dimensional array for logits");
        if (expert_sizes_buf.ndim != 1)
            throw std::runtime_error("Expected a 1-dimensional array for expertKernelSizes");
        if (mask_buf.ndim != 2)
            throw std::runtime_error("Expected a 2-dimensional array for attentionMask");
        int B = logits_buf.shape[0];
        int L = logits_buf.shape[1];
        int N = logits_buf.shape[2];
        if (B != mask_buf.shape[0])
            throw std::runtime_error("Batch size mismatch between logits and attentionMask");
        if (L != mask_buf.shape[1])
            throw std::runtime_error("Sequence length mismatch between logits and attentionMask");
        if (N != expert_sizes_buf.shape[0])
            throw std::runtime_error("Number of experts mismatch between logits and expertKernelSizes");

        // Initialize the logits matrix with direct memory copying
        std::vector<std::vector<std::vector<float>>> logits(B, std::vector<std::vector<float>>(L, std::vector<float>(N)));
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                std::memcpy(logits[b][l].data(), reinterpret_cast<float*>(logits_buf.ptr) + b * logits_buf.strides[0] / sizeof(float) + l * logits_buf.strides[1] / sizeof(float), N * sizeof(float));
            }
        }

        // Initialize the expert kernel sizes with direct memory copying
        std::vector<int> expert_kernel_sizes(N);
        std::memcpy(expert_kernel_sizes.data(), expert_sizes_buf.ptr, N * sizeof(int));

        // Initialize the attention mask with direct memory copying
        std::vector<std::vector<int>> attention_mask(B, std::vector<int>(L));
        for (int b = 0; b < B; ++b) {
            std::memcpy(attention_mask[b].data(), reinterpret_cast<int*>(mask_buf.ptr) + b * mask_buf.strides[0] / sizeof(int), L * sizeof(int));
        }

        // Assume BasicUnitNMSVector is some class that does the actual processing
        BasicUnitNMSVector BasicUnitNMSVectorClass(logits, expert_kernel_sizes, attention_mask);
        auto [center, all] = BasicUnitNMSVectorClass.batched_basic_unit_nms_sorted();

        // Minus 1 to match the original indexing for the returned arrays (Zero-based indexing for experts, -1 for no expert)
        for (int b = 0; b < B; ++b) {
            for (int l = 0; l < L; ++l) {
                center[b][l] -= 1;
                all[b][l] -= 1;
            }
        }

        // Convert the resulting center and all vectors back to NumPy arrays
        py::array_t<int> center_arr = py::array_t<int>({B, L});
        py::array_t<int> all_arr = py::array_t<int>({B, L});

        py::buffer_info center_buf = center_arr.request();
        py::buffer_info all_buf = all_arr.request();

        for (int b = 0; b < B; ++b) {
            std::memcpy(reinterpret_cast<int*>(center_buf.ptr) + b * center_buf.strides[0] / sizeof(int), center[b].data(), L * sizeof(int));
            std::memcpy(reinterpret_cast<int*>(all_buf.ptr) + b * all_buf.strides[0] / sizeof(int), all[b].data(), L * sizeof(int));
        }

        return {center_arr, all_arr};
}

PYBIND11_MODULE(BasicUnitNMS, m) {
    m.def("basic_unit_nms", &basic_unit_nms, "Generate basic_unit masks by applying non-maximum suppression to the logits");
}
