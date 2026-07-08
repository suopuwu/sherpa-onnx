// sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/fast-clustering.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-impl.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

namespace {  // NOLINT

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L41
template <class T>
inline void hash_combine(std::size_t *seed, const T &v) {  // NOLINT
  std::hash<T> hasher;
  *seed ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);  // NOLINT
}

// copied from https://github.com/k2-fsa/k2/blob/master/k2/csrc/host/util.h#L47
struct PairHash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    std::size_t result = 0;
    hash_combine(&result, pair.first);
    hash_combine(&result, pair.second);
    return result;
  }
};
}  // namespace

using Matrix2D = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;  // NOLINT

using Matrix2DInt32 = Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>;  // NOLINT

using FloatRowVector = Eigen::Matrix<float, 1, Eigen::Dynamic>;
using Int32RowVector = Eigen::Matrix<int32_t, 1, Eigen::Dynamic>;

using Int32Pair = std::pair<int32_t, int32_t>;

class OfflineSpeakerDiarizationPyannoteImpl
    : public OfflineSpeakerDiarizationImpl {
 public:
  ~OfflineSpeakerDiarizationPyannoteImpl() override = default;

  explicit OfflineSpeakerDiarizationPyannoteImpl(
      const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(config_.segmentation),
        embedding_extractor_(config_.embedding),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    Init();
  }

  template <typename Manager>
  OfflineSpeakerDiarizationPyannoteImpl(
      Manager *mgr, const OfflineSpeakerDiarizationConfig &config)
      : config_(config),
        segmentation_model_(mgr, config_.segmentation),
        embedding_extractor_(mgr, config_.embedding),
        clustering_(std::make_unique<FastClustering>(config_.clustering)) {
    Init();
  }

  int32_t SampleRate() const override {
    const auto &meta_data = segmentation_model_.GetModelMetaData();

    return meta_data.sample_rate;
  }

  void SetConfig(const OfflineSpeakerDiarizationConfig &config) override {
    if (!config.clustering.Validate()) {
      SHERPA_ONNX_LOGE("Invalid clustering config. Skip it");
      return;
    }
    clustering_ = std::make_unique<FastClustering>(config.clustering);
    config_.clustering = config.clustering;
  }

  OfflineSpeakerDiarizationResult Process(
      const float *audio, int32_t n,
      OfflineSpeakerDiarizationProgressCallback callback = nullptr,
      void *callback_arg = nullptr) const override {
    // suopTranscriber local patch: THREE phases now all report real
    // progress, blended into one combined 0..kProgressScale counter —
    // callers (see DiarizationEngine::diarize()'s onProgress) don't need
    // to know anything changed: numTotalChunks stays a constant
    // kProgressScale throughout, numProcessedChunks just grows
    // continuously across all three instead of sitting frozen through
    // segmentation (upstream NEVER reported progress there), through label
    // conversion just below (plain CPU work, also never reported — this
    // one only became noticeable once the other two gaps were fixed:
    // measured a 6.6s silent stretch here on a real recording), and
    // through embedding (see ComputeEmbeddings()'s doc comment for how
    // batching broke ITS progress specifically). The shares are a fixed
    // approximation of each phase's typical time (measured roughly
    // 28/2/70 on a real recording) — not adaptive per-file, just enough to
    // keep the bar moving smoothly throughout.
    auto segmentationProgress = [callback, callback_arg](int32_t done, int32_t total) {
      if (callback && total > 0) {
        callback(static_cast<int32_t>(static_cast<double>(done) * kSegmentationProgressShare *
                                       kProgressScale / total),
                 kProgressScale, callback_arg);
      }
    };

    std::vector<Matrix2D> segmentations =
        RunSpeakerSegmentationModel(audio, n, segmentationProgress);
    // segmentations[i] is for chunk_i
    // Each matrix is of shape (num_frames, num_powerset_classes)
    if (segmentations.empty()) {
      return {};
    }

    std::vector<Matrix2DInt32> labels;
    labels.reserve(segmentations.size());

    {
      const int32_t total_windows = static_cast<int32_t>(segmentations.size());
      const int32_t base = static_cast<int32_t>(kSegmentationProgressShare * kProgressScale);
      const int32_t range = static_cast<int32_t>(kLabelConversionProgressShare * kProgressScale);
      int32_t label_index = 0;
      for (const auto &m : segmentations) {
        labels.push_back(ToMultiLabel(m));
        ++label_index;
        if (callback && total_windows > 0) {
          callback(base + static_cast<int32_t>(static_cast<double>(label_index) * range / total_windows),
                   kProgressScale, callback_arg);
        }
      }
    }

    segmentations.clear();

    if (labels.size() == 1) {
      if (callback) {
        callback(kProgressScale, kProgressScale, callback_arg);
      }

      return HandleOneChunkSpecialCase(labels[0], n);
    }

    // labels[i] is a 0-1 matrix of shape (num_frames, num_speakers)

    // speaker count per frame
    Int32RowVector speakers_per_frame = ComputeSpeakersPerFrame(labels);

    if (speakers_per_frame.maxCoeff() == 0) {
      SHERPA_ONNX_LOGE("No speakers found in the audio samples");
      return {};
    }

    auto chunk_speaker_samples_list_pair = GetChunkSpeakerSampleIndexes(labels);

    // The embedding model may output NaN. valid_indexes contains indexes
    // in chunk_speaker_samples_list_pair.second that don't lead to
    // NaN embeddings.
    std::vector<int32_t> valid_indexes;
    valid_indexes.reserve(chunk_speaker_samples_list_pair.second.size());

    const int32_t embeddingProgressBase = static_cast<int32_t>(
        (kSegmentationProgressShare + kLabelConversionProgressShare) * kProgressScale);
    if (callback) {
      callback(embeddingProgressBase, kProgressScale, callback_arg);
    }

    auto embeddingProgress = [callback, callback_arg, embeddingProgressBase](int32_t done, int32_t total) {
      if (callback && total > 0) {
        const int32_t range = kProgressScale - embeddingProgressBase;
        callback(embeddingProgressBase + static_cast<int32_t>(static_cast<double>(done) * range / total),
                 kProgressScale, callback_arg);
      }
    };

    Matrix2D embeddings =
        ComputeEmbeddings(audio, n, chunk_speaker_samples_list_pair.second,
                          &valid_indexes, embeddingProgress);

    if (valid_indexes.size() != chunk_speaker_samples_list_pair.second.size()) {
      std::vector<Int32Pair> chunk_speaker_pair;
      std::vector<std::vector<Int32Pair>> sample_indexes;

      chunk_speaker_pair.reserve(valid_indexes.size());
      sample_indexes.reserve(valid_indexes.size());
      for (auto i : valid_indexes) {
        if (i < 0 ||
            i >= static_cast<int32_t>(
                     chunk_speaker_samples_list_pair.first.size())) {
          SHERPA_ONNX_LOGE("valid_indexes: index %d out of range [0, %d)",
                           i,
                           static_cast<int32_t>(
                               chunk_speaker_samples_list_pair.first.size()));
          continue;
        }
        chunk_speaker_pair.push_back(chunk_speaker_samples_list_pair.first[i]);
        sample_indexes.push_back(
            std::move(chunk_speaker_samples_list_pair.second[i]));
      }

      chunk_speaker_samples_list_pair.first = std::move(chunk_speaker_pair);
      chunk_speaker_samples_list_pair.second = std::move(sample_indexes);
    }

    std::vector<int32_t> cluster_labels = clustering_->Cluster(
        &embeddings(0, 0), embeddings.rows(), embeddings.cols());

    if (cluster_labels.empty()) {
      SHERPA_ONNX_LOGE("No speakers found in the audio samples");
      return {};
    }

    int32_t max_cluster_index =
        *std::max_element(cluster_labels.begin(), cluster_labels.end());

    auto chunk_speaker_to_cluster = ConvertChunkSpeakerToCluster(
        chunk_speaker_samples_list_pair.first, cluster_labels);

    auto new_labels =
        ReLabel(labels, max_cluster_index, chunk_speaker_to_cluster);

    Matrix2DInt32 speaker_count = ComputeSpeakerCount(new_labels, n);

    Matrix2DInt32 final_labels =
        FinalizeLabels(speaker_count, speakers_per_frame);

    auto result = ComputeResult(final_labels);

    return result;
  }

 private:
  void Init() { InitPowersetMapping(); }

  // see also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/utils/powerset.py#L68
  void InitPowersetMapping() {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t num_classes = meta_data.num_classes;
    int32_t powerset_max_classes = meta_data.powerset_max_classes;
    int32_t num_speakers = meta_data.num_speakers;

    powerset_mapping_ = Matrix2DInt32(num_classes, num_speakers);
    powerset_mapping_.setZero();

    int32_t k = 1;
    for (int32_t i = 1; i <= powerset_max_classes; ++i) {
      if (i == 1) {
        for (int32_t j = 0; j != num_speakers; ++j, ++k) {
          powerset_mapping_(k, j) = 1;
        }
      } else if (i == 2) {
        for (int32_t j = 0; j != num_speakers; ++j) {
          for (int32_t m = j + 1; m < num_speakers; ++m, ++k) {
            powerset_mapping_(k, j) = 1;
            powerset_mapping_(k, m) = 1;
          }
        }
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %{public}d is currently not supported!", i);
#else
        SHERPA_ONNX_LOGE(
            "powerset_max_classes = %d is currently not supported!", i);
#endif
        SHERPA_ONNX_EXIT(-1);
      }
    }
  }

  // suopTranscriber local patch: how many sliding windows to batch into a
  // single Ort::Session::Run() call for the segmentation model — see
  // RunSpeakerSegmentationModel()'s doc comment for why. Not tuned per-GPU;
  // a size/memory compromise: big enough to meaningfully cut the call count
  // (hundreds of windows -> tens of batches for a real recording), small
  // enough that batch_count * window_size floats (~20 MB at 32 windows *
  // 160000 samples/window * 4 bytes, pyannote's actual 10s-at-16kHz window)
  // stays a trivial, one-off allocation regardless of provider.
  static constexpr int32_t kSegmentationBatchSize = 32;

  // suopTranscriber local patch: see Process()'s doc comment for the
  // combined-progress design these feed — segmentation batches get
  // kSegmentationProgressShare, the label-conversion loop right after gets
  // kLabelConversionProgressShare, and embedding gets everything else
  // (1 - the other two).
  static constexpr int32_t kProgressScale = 1000;
  static constexpr double kSegmentationProgressShare = 0.25;
  static constexpr double kLabelConversionProgressShare = 0.05;

  // suopTranscriber local patch: see ComputeEmbeddings()'s doc comment.
  // Bounds how many OnlineStream objects (each retaining its own extracted
  // fbank features — num_frames * 80 floats, not freed until consumed) are
  // held in memory at once, instead of building one per speech segment for
  // the WHOLE file up front. 512 is 16x kEmbeddingBatchSize — big enough
  // that length-bucketing (see ComputeBatch()) still has a decent pool of
  // candidates to group similar-length segments from, small enough that
  // peak memory here stays a bounded constant regardless of file length —
  // not deeply tuned.
  static constexpr int32_t kEmbeddingStreamWindowSize = 512;

  std::vector<Matrix2D> RunSpeakerSegmentationModel(
      const float *audio, int32_t n,
      const std::function<void(int32_t, int32_t)> &progressCallback = {}) const {
    std::vector<Matrix2D> ans;

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;

    if (n <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "number of audio samples is %{public}d (<= 0). Please provide a "
          "positive number",
          n);
#else
      SHERPA_ONNX_LOGE(
          "number of audio samples is %d (<= 0). Please provide a positive "
          "number",
          n);
#endif
      return {};
    }

    if (n <= window_size) {
      std::vector<float> buf(window_size);
      // NOTE: buf is zero initialized by default

      std::copy(audio, audio + n, buf.data());

      Matrix2D m = ProcessChunk(buf.data());

      ans.push_back(std::move(m));

      if (progressCallback) {
        progressCallback(1, 1);
      }

      return ans;
    }

    int32_t num_chunks = (n - window_size) / window_shift + 1;
    bool has_last_chunk = ((n - window_size) % window_shift) > 0;

    ans.reserve(num_chunks + has_last_chunk);

    // suopTranscriber local patch (see patches/sherpa-onnx-batch-
    // segmentation.patch): batch multiple sliding windows into ONE
    // Ort::Session::Run() call instead of one call per window (the
    // upstream default just below, still used for the <=1 chunk and
    // has_last_chunk tail cases). Each window is a small, cheap forward
    // pass; under CUDA specifically, kernel-launch/host-device-sync
    // overhead between calls dominates over actual compute for a real
    // recording's hundreds of sequential windows (measured: ~10% GPU
    // utilization unbatched), so batching amortizes that per-call
    // overhead across real work instead of paying it once per window.
    //
    // total_batches also feeds progressCallback below — upstream never
    // reported segmentation progress at all; see Process()'s doc comment
    // for why that changed here.
    const int32_t total_batches =
        (num_chunks + kSegmentationBatchSize - 1) / kSegmentationBatchSize +
        (has_last_chunk ? 1 : 0);
    int32_t completed_batches = 0;
    for (int32_t start = 0; start < num_chunks; start += kSegmentationBatchSize) {
      int32_t batch_count = std::min(kSegmentationBatchSize, num_chunks - start);
      std::vector<Matrix2D> batch = ProcessBatch(
          audio + static_cast<int64_t>(start) * window_shift, window_shift,
          window_size, batch_count);
      for (Matrix2D &m : batch) {
        ans.push_back(std::move(m));
      }
      ++completed_batches;
      if (progressCallback) {
        progressCallback(completed_batches, total_batches);
      }
    }

    if (has_last_chunk) {
      const float *p = audio + static_cast<int64_t>(num_chunks) * window_shift;
      std::vector<float> buf(window_size);
      std::copy(p, audio + n, buf.data());

      Matrix2D m = ProcessChunk(buf.data());

      ans.push_back(std::move(m));

      ++completed_batches;
      if (progressCallback) {
        progressCallback(completed_batches, total_batches);
      }
    }

    return ans;
  }

  Matrix2D ProcessChunk(const float *p) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> shape = {1, 1, window_size};

    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, const_cast<float *>(p),
                                 window_size, shape.data(), shape.size());

    Ort::Value out = segmentation_model_.Forward(std::move(x));
    std::vector<int64_t> out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    Matrix2D m(out_shape[1], out_shape[2]);
    std::copy(out.GetTensorData<float>(), out.GetTensorData<float>() + m.size(),
              &m(0, 0));
    return m;
  }

  // suopTranscriber local patch: see RunSpeakerSegmentationModel()'s call
  // site comment above. `p` points `batch_count` sliding windows of
  // `window_size` samples each (spaced `window_shift` samples apart,
  // OVERLAPPING when window_shift < window_size, as pyannote's 90% overlap
  // does) into the caller's audio buffer — all `batch_count` windows are
  // guaranteed fully in-bounds by construction (only called for the
  // num_chunks range; the zero-padded final partial window still goes
  // through the original single-chunk ProcessChunk() above).
  std::vector<Matrix2D> ProcessBatch(const float *p, int32_t window_shift,
                                      int32_t window_size,
                                      int32_t batch_count) const {
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    // Windows overlap, so they can't be read as one contiguous
    // [batch_count, window_size] block of the ORIGINAL audio buffer
    // directly — each window is copied into its own slot of a fresh,
    // truly contiguous buffer instead. A memcpy is trivially cheap next
    // to the inference call this feeds.
    std::vector<float> buf(static_cast<size_t>(batch_count) * window_size);
    for (int32_t i = 0; i != batch_count; ++i) {
      std::copy(p + static_cast<int64_t>(i) * window_shift,
                p + static_cast<int64_t>(i) * window_shift + window_size,
                buf.data() + static_cast<size_t>(i) * window_size);
    }

    std::array<int64_t, 3> shape = {batch_count, 1, window_size};

    Ort::Value x = Ort::Value::CreateTensor(memory_info, buf.data(),
                                            buf.size(), shape.data(),
                                            shape.size());

    Ort::Value out = segmentation_model_.Forward(std::move(x));
    std::vector<int64_t> out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
    // out_shape = {batch_count, num_frames, num_classes} — the model's
    // batch axis is dynamic (confirmed against the actual .onnx graph,
    // not just the Forward() doc comment), so this just works.
    int32_t num_frames = out_shape[1];
    int32_t num_classes = out_shape[2];
    const float *out_data = out.GetTensorData<float>();

    std::vector<Matrix2D> ans;
    ans.reserve(batch_count);
    for (int32_t i = 0; i != batch_count; ++i) {
      Matrix2D m(num_frames, num_classes);
      std::copy(
          out_data + static_cast<size_t>(i) * num_frames * num_classes,
          out_data + static_cast<size_t>(i + 1) * num_frames * num_classes,
          &m(0, 0));
      ans.push_back(std::move(m));
    }
    return ans;
  }

  Matrix2DInt32 ToMultiLabel(const Matrix2D &m) const {
    int32_t num_rows = m.rows();
    Matrix2DInt32 ans(num_rows, powerset_mapping_.cols());

    std::ptrdiff_t col_id;

    for (int32_t i = 0; i != num_rows; ++i) {
      m.row(i).maxCoeff(&col_id);
      ans.row(i) = powerset_mapping_.row(col_id);
    }

    return ans;
  }

  // See also
  // https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/utils/diarization.py#L122
  Int32RowVector ComputeSpeakersPerFrame(
      const std::vector<Matrix2DInt32> &labels) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;

    FloatRowVector count(num_frames);
    FloatRowVector weight(num_frames);
    count.setZero();
    weight.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq).array() += labels[i].rowwise().sum().array().cast<float>();

      weight(seq).array() += 1;
    }

    return ((count.array() / (weight.array() + 1e-12f)) + 0.5).cast<int32_t>();
  }

  // ans.first: a list of (chunk_id, speaker_id)
  // ans.second: a list of list of (start_sample_index, end_sample_index)
  //
  // ans.first[i] corresponds to ans.second[i]
  std::pair<std::vector<Int32Pair>, std::vector<std::vector<Int32Pair>>>
  GetChunkSpeakerSampleIndexes(const std::vector<Matrix2DInt32> &labels) const {
    auto new_labels = ExcludeOverlap(labels);

    std::vector<Int32Pair> chunk_speaker_list;
    std::vector<std::vector<Int32Pair>> samples_index_list;

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t num_speakers = meta_data.num_speakers;

    int32_t chunk_index = 0;
    for (const auto &label : new_labels) {
      Matrix2DInt32 tmp = label.transpose();
      // tmp: (num_speakers, num_frames)
      int32_t num_frames = tmp.cols();

      int32_t sample_offset = chunk_index * window_shift;

      for (int32_t speaker_index = 0; speaker_index != num_speakers;
           ++speaker_index) {
        auto d = tmp.row(speaker_index);
        if (d.sum() < 10) {
          // skip segments less than 10 frames
          continue;
        }

        Int32Pair this_chunk_speaker = {chunk_index, speaker_index};
        std::vector<Int32Pair> this_speaker_samples;

        bool is_active = false;
        int32_t start_index;

        for (int32_t k = 0; k != num_frames; ++k) {
          if (d[k] != 0) {
            if (!is_active) {
              is_active = true;
              start_index = k;
            }
          } else if (is_active) {
            is_active = false;

            int32_t start_samples =
                static_cast<float>(start_index) / num_frames * window_size +
                sample_offset;
            int32_t end_samples =
                static_cast<float>(k) / num_frames * window_size +
                sample_offset;

            this_speaker_samples.emplace_back(start_samples, end_samples);
          }
        }

        if (is_active) {
          int32_t start_samples =
              static_cast<float>(start_index) / num_frames * window_size +
              sample_offset;
          int32_t end_samples =
              static_cast<float>(num_frames - 1) / num_frames * window_size +
              sample_offset;
          this_speaker_samples.emplace_back(start_samples, end_samples);
        }

        chunk_speaker_list.push_back(std::move(this_chunk_speaker));
        samples_index_list.push_back(std::move(this_speaker_samples));
      }  // for (int32_t speaker_index = 0;
      chunk_index += 1;
    }  // for (const auto &label : new_labels)

    return {chunk_speaker_list, samples_index_list};
  }

  // If there are multiple speakers at a frame, then this frame is excluded.
  std::vector<Matrix2DInt32> ExcludeOverlap(
      const std::vector<Matrix2DInt32> &labels) const {
    int32_t num_chunks = labels.size();
    std::vector<Matrix2DInt32> ans;
    ans.reserve(num_chunks);

    for (const auto &label : labels) {
      Matrix2DInt32 new_label(label.rows(), label.cols());
      new_label.setZero();
      Int32RowVector v = label.rowwise().sum();

      for (int32_t i = 0; i != v.cols(); ++i) {
        if (v[i] < 2) {
          new_label.row(i) = label.row(i);
        }
      }

      ans.push_back(std::move(new_label));
    }

    return ans;
  }

  /**
   * @param sample_indexes[i] contains the sample segment start and end indexes
   *                          for the i-th (chunk, speaker) pair
   * @return Return a matrix of shape (sample_indexes.size(), embedding_dim)
   *         where ans.row[i] contains the embedding for the
   *         i-th (chunk, speaker) pair
   */
  // suopTranscriber local patch: progressCallback is a simple (done, total)
  // pair now — not the raw OfflineSpeakerDiarizationProgressCallback C-style
  // pair — because ComputeBatch() below reports progress ONCE PER COMPLETED
  // BATCH (a whole bucket finishing together, not one item at a time), so
  // this function no longer needs its own per-item callback loop at all
  // (removed — see below); it just forwards Process()'s already-scaled
  // lambda straight through to ComputeBatch().
  Matrix2D ComputeEmbeddings(
      const float *audio, int32_t n,
      const std::vector<std::vector<Int32Pair>> &sample_indexes,
      std::vector<int32_t> *valid_indexes,
      const std::function<void(int32_t, int32_t)> &progressCallback) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t sample_rate = meta_data.sample_rate;
    Matrix2D ans(sample_indexes.size(), embedding_extractor_.Dim());

    auto IsNaNWrapper = [](float f) -> bool { return std::isnan(f); };

    // suopTranscriber local patch (see patches/sherpa-onnx-batch-
    // segmentation.patch): build a WINDOW of streams at a time (cheap CPU
    // feature extraction — must stay sequential, each needs its own
    // OnlineStream), THEN batch the embedding-model inference across that
    // window in one embedding_extractor_.ComputeBatch() call instead of
    // one embedding_extractor_.Compute() call per stream — see
    // SpeakerEmbeddingExtractorGeneralImpl::ComputeBatch()'s doc comment
    // for why this one (unlike segmentation) needs length-bucketing.
    //
    // Windowed, NOT "build every stream for the whole file up front": each
    // OnlineStream retains its extracted fbank features (num_frames * 80
    // floats) until consumed, so holding ALL of them simultaneously for a
    // long recording scales memory with TOTAL FILE LENGTH — a real,
    // multi-gigabyte-scale risk for a many-hour recording with thousands
    // of speech segments (an earlier version of this patch did exactly
    // that, and a clean std::bad_alloc there gets silently caught by
    // DiarizationEngine::diarize()'s own try/catch, see its doc comment,
    // degrading to "no speakers found" with NO visible error — exactly the
    // "speakers silently stopped being written on large files" symptom
    // this fix addresses). kEmbeddingStreamWindowSize bounds peak memory
    // to a constant regardless of file length.
    //
    // AcceptWaveform() below is where the actual fbank feature extraction
    // happens (ComputeBatch()'s own "Phase 1" just RETRIEVES already-
    // computed frames — cheap; this loop is where the real, sometimes
    // lengthy, CPU cost of the whole embedding phase actually is, measured
    // directly: a 6.6s silent gap on a real 24-minute recording before
    // progress reporting was added here). Reported as the first half of
    // this call's own progress range; ComputeBatch()'s own report becomes
    // the second half via computeBatchProgress below — Process()'s
    // embeddingProgress lambda doesn't need to know any of this, it just
    // keeps rescaling whatever (done, total) pair arrives here into its
    // own [base, kProgressScale] slice.
    const int32_t total_items = static_cast<int32_t>(sample_indexes.size());
    const int32_t total_units = total_items * 2;

    int32_t k = 0;
    int32_t cur_row_index = 0;

    for (size_t window_start = 0; window_start < sample_indexes.size();
         window_start += kEmbeddingStreamWindowSize) {
      const size_t window_end = std::min(
          sample_indexes.size(), window_start + static_cast<size_t>(kEmbeddingStreamWindowSize));
      const int32_t window_size = static_cast<int32_t>(window_end - window_start);

      std::vector<std::unique_ptr<OnlineStream>> streams;
      streams.reserve(window_size);
      std::vector<OnlineStream *> stream_ptrs;
      stream_ptrs.reserve(window_size);

      for (size_t i = window_start; i != window_end; ++i) {
        const auto &v = sample_indexes[i];
        auto stream = embedding_extractor_.CreateStream();
        for (const auto &p : v) {
          int32_t end = (p.second <= n) ? p.second : n;
          int32_t num_samples = end - p.first;

          if (num_samples > 0) {
            stream->AcceptWaveform(sample_rate, audio + p.first, num_samples);
          }
        }

        stream->InputFinished();
        if (!embedding_extractor_.IsReady(stream.get())) {
          SHERPA_ONNX_LOGE(
              "This segment is too short, which should not happen since we have "
              "already filtered short segments");
          SHERPA_ONNX_EXIT(-1);
        }

        stream_ptrs.push_back(stream.get());
        streams.push_back(std::move(stream));

        // suopTranscriber local patch: window_start IS the count of items
        // fully accounted for by all PRIOR windows (windows are
        // contiguous), so window_start*2 is exactly where this window's
        // own [build, compute] unit range begins within total_units.
        // Fixed a real bug here: an earlier version used total_items
        // (not window_start) as this window's base, which only made sense
        // under the OLD "build every stream for the whole file, THEN
        // compute" design — under windowed processing (build window 1,
        // compute window 1, build window 2, ...) that made progress jump
        // forward to ~50% after window 1's compute phase, then jump
        // BACKWARD when window 2's build phase resumed counting from a
        // much lower absolute position — reported directly: "jumps back
        // and forth between 20 and 40 percent".
        if (progressCallback) {
          const int32_t local_position = static_cast<int32_t>(i - window_start);
          progressCallback(static_cast<int32_t>(window_start) * 2 + local_position + 1, total_units);
        }
      }

      auto computeBatchProgress = [&progressCallback, total_units, window_start,
                                   window_size](int32_t done, int32_t total) {
        if (progressCallback && total > 0) {
          const int32_t window_done = static_cast<int32_t>(static_cast<double>(done) * window_size / total);
          const int32_t base = static_cast<int32_t>(window_start) * 2 + window_size;
          progressCallback(base + window_done, total_units);
        }
      };

      std::vector<std::vector<float>> window_embeddings =
          embedding_extractor_.ComputeBatch(stream_ptrs, computeBatchProgress);

      for (const std::vector<float> &embedding : window_embeddings) {
        if (std::none_of(embedding.begin(), embedding.end(), IsNaNWrapper)) {
          // a valid embedding
          std::copy(embedding.begin(), embedding.end(), &ans(cur_row_index, 0));
          cur_row_index += 1;
          valid_indexes->push_back(k);
        }

        k += 1;
      }
      // `streams` (and every OnlineStream's fbank feature memory) is freed
      // here, at the end of this window's scope, before the next window's
      // streams are built.
    }

    if (k != cur_row_index) {
      auto seq = Eigen::seqN(0, cur_row_index);
      ans = ans(seq, Eigen::placeholders::all);
    }

    return ans;
  }

  std::unordered_map<Int32Pair, int32_t, PairHash> ConvertChunkSpeakerToCluster(
      const std::vector<Int32Pair> &chunk_speaker_pair,
      const std::vector<int32_t> &cluster_labels) const {
    std::unordered_map<Int32Pair, int32_t, PairHash> ans;

    int32_t k = 0;
    for (const auto &p : chunk_speaker_pair) {
      if (k >= static_cast<int32_t>(cluster_labels.size())) {
        SHERPA_ONNX_LOGE("cluster_labels size mismatch: k=%d, size=%d", k,
                         static_cast<int32_t>(cluster_labels.size()));
        break;
      }
      ans[p] = cluster_labels[k];
      k += 1;
    }

    return ans;
  }

  std::vector<Matrix2DInt32> ReLabel(
      const std::vector<Matrix2DInt32> &labels, int32_t max_cluster_index,
      std::unordered_map<Int32Pair, int32_t, PairHash> chunk_speaker_to_cluster)
      const {
    std::vector<Matrix2DInt32> new_labels;
    new_labels.reserve(labels.size());

    int32_t chunk_index = 0;
    for (const auto &label : labels) {
      Matrix2DInt32 new_label(label.rows(), max_cluster_index + 1);
      new_label.setZero();

      Matrix2DInt32 t = label.transpose();
      // t: (num_speakers, num_frames)

      for (int32_t speaker_index = 0; speaker_index != t.rows();
           ++speaker_index) {
        if (chunk_speaker_to_cluster.count({chunk_index, speaker_index}) == 0) {
          continue;
        }

        int32_t new_speaker_index =
            chunk_speaker_to_cluster.at({chunk_index, speaker_index});

        if (new_speaker_index < 0 ||
            new_speaker_index >= new_label.cols()) {
          SHERPA_ONNX_LOGE(
              "ReLabel: new_speaker_index %d out of range [0, %d), skipping",
              new_speaker_index,
              static_cast<int32_t>(new_label.cols()));
          continue;
        }

        for (int32_t k = 0; k != t.cols(); ++k) {
          if (t(speaker_index, k) == 1) {
            new_label(k, new_speaker_index) = 1;
          }
        }
      }

      new_labels.push_back(std::move(new_label));

      chunk_index += 1;
    }

    return new_labels;
  }

  Matrix2DInt32 ComputeSpeakerCount(const std::vector<Matrix2DInt32> &labels,
                                    int32_t num_samples) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    int32_t num_chunks = labels.size();

    int32_t num_frames = (window_size + (num_chunks - 1) * window_shift) /
                             receptive_field_shift +
                         1;

    Matrix2DInt32 count(num_frames, labels[0].cols());
    count.setZero();

    for (int32_t i = 0; i != num_chunks; ++i) {
      int32_t start =
          static_cast<float>(i) * window_shift / receptive_field_shift + 0.5;

      auto seq = Eigen::seqN(start, labels[i].rows());

      count(seq, Eigen::placeholders::all).array() += labels[i].array();
    }

    bool has_last_chunk = ((num_samples - window_size) % window_shift) > 0;

    if (!has_last_chunk) {
      return count;
    }

    int32_t last_frame = num_samples / receptive_field_shift;
    if (last_frame >= count.rows()) {
      last_frame = count.rows() - 1;
    }
    return count(Eigen::seqN(0, last_frame + 1), Eigen::placeholders::all);
  }

  Matrix2DInt32 FinalizeLabels(const Matrix2DInt32 &count,
                               const Int32RowVector &speakers_per_frame) const {
    int32_t num_rows = count.rows();
    int32_t num_cols = count.cols();

    Matrix2DInt32 ans(num_rows, num_cols);
    ans.setZero();

    for (int32_t i = 0; i != num_rows; ++i) {
      int32_t k = speakers_per_frame[i];
      if (k == 0) {
        continue;
      }
      auto top_k = TopkIndex(&count(i, 0), num_cols, k);

      for (int32_t m : top_k) {
        if (m >= 0 && m < num_cols) {
          ans(i, m) = 1;
        }
      }
    }

    return ans;
  }

  OfflineSpeakerDiarizationResult ComputeResult(
      const Matrix2DInt32 &final_labels) const {
    Matrix2DInt32 final_labels_t = final_labels.transpose();
    int32_t num_speakers = final_labels_t.rows();
    int32_t num_frames = final_labels_t.cols();

    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;
    int32_t receptive_field_size = meta_data.receptive_field_size;
    int32_t sample_rate = meta_data.sample_rate;

    float scale = static_cast<float>(receptive_field_shift) / sample_rate;
    float scale_offset = 0.5 * receptive_field_size / sample_rate;

    OfflineSpeakerDiarizationResult ans;

    for (int32_t speaker_index = 0; speaker_index != num_speakers;
         ++speaker_index) {
      std::vector<OfflineSpeakerDiarizationSegment> this_speaker;

      bool is_active = final_labels_t(speaker_index, 0) > 0;
      int32_t start_index = is_active ? 0 : -1;

      for (int32_t frame_index = 1; frame_index != num_frames; ++frame_index) {
        if (is_active) {
          if (final_labels_t(speaker_index, frame_index) == 0) {
            float start_time = start_index * scale + scale_offset;
            float end_time = frame_index * scale + scale_offset;

            OfflineSpeakerDiarizationSegment segment(start_time, end_time,
                                                     speaker_index);
            this_speaker.push_back(segment);

            is_active = false;
          }
        } else if (final_labels_t(speaker_index, frame_index) == 1) {
          is_active = true;
          start_index = frame_index;
        }
      }

      if (is_active) {
        float start_time = start_index * scale + scale_offset;
        float end_time = (num_frames - 1) * scale + scale_offset;

        OfflineSpeakerDiarizationSegment segment(start_time, end_time,
                                                 speaker_index);
        this_speaker.push_back(segment);
      }

      // merge segments if the gap between them is less than min_duration_off
      MergeSegments(&this_speaker);

      for (const auto &seg : this_speaker) {
        if (seg.Duration() > config_.min_duration_on) {
          ans.Add(seg);
        }
      }
    }  // for (int32_t speaker_index = 0; speaker_index != num_speakers;

    return ans;
  }

  OfflineSpeakerDiarizationResult HandleOneChunkSpecialCase(
      const Matrix2DInt32 &final_labels, int32_t num_samples) const {
    const auto &meta_data = segmentation_model_.GetModelMetaData();
    int32_t window_size = meta_data.window_size;
    int32_t window_shift = meta_data.window_shift;
    int32_t receptive_field_shift = meta_data.receptive_field_shift;

    bool has_last_chunk = (num_samples - window_size) % window_shift > 0;
    if (!has_last_chunk) {
      return ComputeResult(final_labels);
    }

    int32_t num_frames = final_labels.rows();

    int32_t new_num_frames = num_samples / receptive_field_shift;

    int32_t num_frames_to_keep = std::min(new_num_frames, num_frames);

    return ComputeResult(
        final_labels(Eigen::seqN(0, num_frames_to_keep), Eigen::placeholders::all));
  }

  void MergeSegments(
      std::vector<OfflineSpeakerDiarizationSegment> *segments) const {
    float min_duration_off = config_.min_duration_off;
    bool changed = true;
    while (changed) {
      changed = false;
      for (int32_t i = 0; i < static_cast<int32_t>(segments->size()) - 1; ++i) {
        auto s = (*segments)[i].Merge((*segments)[i + 1], min_duration_off);
        if (s) {
          (*segments)[i] = s.value();
          segments->erase(segments->begin() + i + 1);

          changed = true;
          break;
        }
      }
    }
  }

 private:
  OfflineSpeakerDiarizationConfig config_;
  OfflineSpeakerSegmentationPyannoteModel segmentation_model_;
  SpeakerEmbeddingExtractor embedding_extractor_;
  std::unique_ptr<FastClustering> clustering_;
  Matrix2DInt32 powerset_mapping_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_OFFLINE_SPEAKER_DIARIZATION_PYANNOTE_IMPL_H_
