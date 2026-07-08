// sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-impl.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-model.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorGeneralImpl
    : public SpeakerEmbeddingExtractorImpl {
 public:
  explicit SpeakerEmbeddingExtractorGeneralImpl(
      const SpeakerEmbeddingExtractorConfig &config)
      : model_(config) {}

  template <typename Manager>
  SpeakerEmbeddingExtractorGeneralImpl(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : model_(mgr, config) {}

  int32_t Dim() const override { return model_.GetMetaData().output_dim; }

  std::unique_ptr<OnlineStream> CreateStream() const override {
    FeatureExtractorConfig feat_config;
    const auto &meta_data = model_.GetMetaData();
    feat_config.sampling_rate = meta_data.sample_rate;
    feat_config.normalize_samples = meta_data.normalize_samples;

    return std::make_unique<OnlineStream>(feat_config);
  }

  bool IsReady(OnlineStream *s) const override {
    return s->GetNumProcessedFrames() < s->NumFramesReady();
  }

  std::vector<float> Compute(OnlineStream *s) const override {
    int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
    if (num_frames <= 0) {
#if __OHOS__
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %{public}d",
          num_frames);
#else
      SHERPA_ONNX_LOGE(
          "Please make sure IsReady(s) returns true. num_frames: %d",
          num_frames);
#endif
      return {};
    }

    std::vector<float> features =
        s->GetFrames(s->GetNumProcessedFrames(), num_frames);

    s->GetNumProcessedFrames() += num_frames;

    int32_t feat_dim = features.size() / num_frames;

    const auto &meta_data = model_.GetMetaData();
    if (!meta_data.feature_normalize_type.empty()) {
      if (meta_data.feature_normalize_type == "global-mean") {
        SubtractGlobalMean(features.data(), num_frames, feat_dim);
      } else {
#if __OHOS__
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %{public}s",
                         meta_data.feature_normalize_type.c_str());
#else
        SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                         meta_data.feature_normalize_type.c_str());
#endif
        SHERPA_ONNX_EXIT(-1);
      }
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> x_shape{1, num_frames, feat_dim};
    Ort::Value x =
        Ort::Value::CreateTensor(memory_info, features.data(), features.size(),
                                 x_shape.data(), x_shape.size());
    Ort::Value embedding = model_.Compute(std::move(x));
    std::vector<int64_t> embedding_shape =
        embedding.GetTensorTypeAndShapeInfo().GetShape();

    std::vector<float> ans(embedding_shape[1]);
    std::copy(embedding.GetTensorData<float>(),
              embedding.GetTensorData<float>() + ans.size(), ans.begin());

    return ans;
  }

  // suopTranscriber local patch (see patches/sherpa-onnx-batch-
  // segmentation.patch): real batched fast path, overriding the naive
  // per-stream default in SpeakerEmbeddingExtractorImpl. Unlike the
  // segmentation model's FIXED window_size (trivial to batch — see
  // offline-speaker-diarization-pyannote-impl.h's sibling patch), each
  // speech segment here has a DIFFERENT frame count, and this model has no
  // length/mask input — every frame in a batch, real or padded, gets
  // processed identically by the conv+pooling stack, so naively batching
  // very different lengths together would let padding measurably distort
  // the pooled embedding for the shorter items. Bucketing by length (see
  // kEmbeddingLengthBucketMaxRatio) keeps worst-case padding waste bounded
  // instead of eliminating it — a deliberate accuracy/speed tradeoff, not
  // a free lunch; validated empirically (see the patch file's own
  // description) rather than assumed safe.
  std::vector<std::vector<float>> ComputeBatch(
      const std::vector<OnlineStream *> &streams,
      const std::function<void(int32_t, int32_t)> &progressCallback = {}) const override {
    const int32_t num_streams = static_cast<int32_t>(streams.size());
    std::vector<std::vector<float>> ans(num_streams);
    if (num_streams == 0) {
      return ans;
    }

    // suopTranscriber local patch: Phase 1 (below) and Phase 2 (batched
    // inference) each count for half of progressCallback's own range here
    // — feature extraction has no natural "batch" grouping the way
    // inference does, but it's still real, sequential, sometimes lengthy
    // CPU work (measured: a 6.9s silent gap on a real 24-minute recording
    // before this was added) that callers should still see move. This
    // doubled total is purely internal to this call — ComputeEmbeddings()
    // forwards whatever (done, total) pair arrives straight through, it
    // doesn't assume total == streams.size().
    const int32_t progressTotal = num_streams * 2;

    // Phase 1: per-stream feature extraction + normalization — the CHEAP
    // CPU part (measured: not the bottleneck; see the segmentation patch's
    // sibling comment for the same finding on that model). Kept exactly
    // as Compute() above does it, just factored out so the expensive ONNX
    // inference below can be batched across streams instead.
    std::vector<EmbeddingBatchItem> items;
    items.reserve(num_streams);
    int32_t feat_dim = 0;

    for (int32_t i = 0; i != num_streams; ++i) {
      OnlineStream *s = streams[i];
      int32_t num_frames = s->NumFramesReady() - s->GetNumProcessedFrames();
      if (num_frames <= 0) {
#if __OHOS__
        SHERPA_ONNX_LOGE(
            "Please make sure IsReady(s) returns true. num_frames: "
            "%{public}d",
            num_frames);
#else
        SHERPA_ONNX_LOGE(
            "Please make sure IsReady(s) returns true. num_frames: %d",
            num_frames);
#endif
        continue;  // ans[i] stays empty, same shape as Compute()'s
                   // undefined-behavior-avoided-by-caller-contract case
      }

      std::vector<float> features =
          s->GetFrames(s->GetNumProcessedFrames(), num_frames);
      s->GetNumProcessedFrames() += num_frames;

      feat_dim = static_cast<int32_t>(features.size()) / num_frames;

      const auto &meta_data = model_.GetMetaData();
      if (!meta_data.feature_normalize_type.empty()) {
        if (meta_data.feature_normalize_type == "global-mean") {
          SubtractGlobalMean(features.data(), num_frames, feat_dim);
        } else {
#if __OHOS__
          SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %{public}s",
                           meta_data.feature_normalize_type.c_str());
#else
          SHERPA_ONNX_LOGE("Unsupported feature_normalize_type: %s",
                           meta_data.feature_normalize_type.c_str());
#endif
          SHERPA_ONNX_EXIT(-1);
        }
      }

      items.push_back({i, num_frames, std::move(features)});

      if (progressCallback) {
        progressCallback(i + 1, progressTotal);
      }
    }

    if (items.empty()) {
      return ans;
    }

    // Phase 2: bucket by length (ascending), then run ONE
    // Ort::Session::Run() per bucket instead of one per item.
    std::sort(items.begin(), items.end(),
              [](const EmbeddingBatchItem &a, const EmbeddingBatchItem &b) {
                return a.num_frames < b.num_frames;
              });

    size_t i = 0;
    int32_t completed_items = 0;
    while (i != items.size()) {
      size_t j = i + 1;
      const int32_t min_frames = items[i].num_frames;
      const int32_t max_frames_allowed = static_cast<int32_t>(
          min_frames * kEmbeddingLengthBucketMaxRatio);
      while (j != items.size() &&
             (j - i) < static_cast<size_t>(kEmbeddingBatchSize) &&
             items[j].num_frames <= max_frames_allowed) {
        // suopTranscriber local patch: kEmbeddingBatchSize and
        // kEmbeddingLengthBucketMaxRatio above only bound item COUNT and
        // the RELATIVE spread of lengths within a bucket — neither bounds
        // the ABSOLUTE tensor size a bucket can grow to. A real recording
        // can contain a very long continuous segment (e.g. someone
        // talking for minutes); batching kEmbeddingBatchSize of THOSE
        // together produces a batch_count * max_frames product large
        // enough for a deep conv net's internal activation memory to
        // exceed GPU VRAM entirely — measured directly: a 1.1 GB single
        // allocation failure (CUDA OOM) inside ONNX Runtime on a real
        // multi-hour recording. This check stops the bucket from growing
        // once admitting the NEXT item (which, since items are sorted
        // ascending, would also become the bucket's new max_frames) would
        // push batch_count * max_frames past kEmbeddingMaxBatchFrames,
        // regardless of whether the count/ratio limits above were already
        // satisfied. A single item alone is never rejected by this check
        // (the bucket always contains at least items[i]) — same
        // unbounded-length tolerance the ORIGINAL unbatched code already
        // had for one segment at a time; only the MULTIPLICATION by
        // batch_count is new, so only that is capped here.
        const int64_t projected_frames =
            static_cast<int64_t>(j - i + 1) * items[j].num_frames;
        if (projected_frames > kEmbeddingMaxBatchFrames) {
          break;
        }
        ++j;
      }

      ProcessEmbeddingBatch(items, i, j, feat_dim, &ans);
      // suopTranscriber local patch: report progress once per completed
      // BUCKET, not once per item — items within a bucket finish together
      // (one Session::Run() call), so there's no meaningful "per-item"
      // moment to report within a bucket. Starts counting from num_streams
      // (Phase 1's share of progressTotal, already fully reported above)
      // rather than 0.
      completed_items += static_cast<int32_t>(j - i);
      if (progressCallback) {
        progressCallback(num_streams + completed_items, progressTotal);
      }
      i = j;
    }

    return ans;
  }

 private:
  void SubtractGlobalMean(float *p, int32_t num_frames,
                          int32_t feat_dim) const {
    auto m = Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        p, num_frames, feat_dim);

    m = m.rowwise() - m.colwise().mean();
  }

  // suopTranscriber local patch: one item queued for ComputeBatch() below —
  // a class-scope type (not function-local) so ProcessEmbeddingBatch() can
  // share it.
  struct EmbeddingBatchItem {
    int32_t index;  // this item's position in ComputeBatch()'s `streams`/`ans`
    int32_t num_frames;
    std::vector<float> features;  // num_frames * feat_dim, already normalized
  };

  // suopTranscriber local patch: see ComputeBatch()'s doc comment. `items`
  // is sorted ascending by num_frames; [begin, end) is one length bucket.
  // Zero-pads every item in the bucket up to the bucket's own max length
  // (items[end - 1].num_frames, since the range is sorted), runs one
  // batched Session::Run(), then writes each item's own embedding back
  // into `*ans` at its ORIGINAL (pre-sort) index.
  //
  // No fixed batch-size/frame-count cap (see kEmbeddingMaxBatchFrames) can
  // ever be GUARANTEED safe — the model's actual internal activation
  // memory scaling isn't visible from here, and available GPU memory
  // varies with whatever ELSE is using the GPU at the time. Confirmed
  // directly: kEmbeddingMaxBatchFrames=100000 was NOT conservative enough
  // in practice — a real 1.1 GB single-allocation CUDA failure still
  // happened on a real multi-hour recording. So this reacts to an actual
  // failure instead of only trying to predict one: on any exception from
  // the inference call, bisect the batch and retry each half — a smaller
  // batch means a smaller tensor, which either succeeds or gets bisected
  // again down to single items. This means ONE oversized batch degrades
  // gracefully (the rest of the file's diarization still completes)
  // instead of the WHOLE diarize() call failing via
  // DiarizationEngine::diarize()'s top-level try/catch, which previously
  // discarded every other segment's already-computed work too — the exact
  // "no speakers written at all, even for the parts that would have
  // worked fine" symptom this fixes.
  void ProcessEmbeddingBatch(const std::vector<EmbeddingBatchItem> &items,
                             size_t begin, size_t end, int32_t feat_dim,
                             std::vector<std::vector<float>> *ans) const {
    const int32_t batch_count = static_cast<int32_t>(end - begin);
    // suopTranscriber local patch: rounded UP to the nearest
    // kEmbeddingFrameQuantum instead of using the bucket's raw longest
    // item directly. Why: ONNX Runtime's default CUDA memory arena is a
    // CACHING allocator — it holds onto distinctly-sized buffers for
    // reuse rather than freeing them back to the driver between calls.
    // Every bucket here can have a slightly different max_frames (real
    // segment lengths vary continuously), so without quantizing, a long
    // multi-hour recording with thousands of buckets asks the arena to
    // cache thousands of UNIQUELY-sized tensors — confirmed directly: one
    // real 4h48m recording's diarization pass grew the CUDA arena to
    // ~29 GB (nearly the entire card) before the eventual 1.1 GB
    // allocation failure that motivated ProcessEmbeddingBatch()'s retry
    // fallback above. Quantizing means many buckets share the exact same
    // padded shape, so the arena can actually REUSE cached blocks instead
    // of continuously growing. Costs a little extra zero-padding (already
    // tolerated — see kEmbeddingLengthBucketMaxRatio), not tuned to a
    // precise memory budget.
    const int32_t max_frames =
        ((items[end - 1].num_frames + kEmbeddingFrameQuantum - 1) / kEmbeddingFrameQuantum) *
        kEmbeddingFrameQuantum;

    try {
      std::vector<float> buf(
          static_cast<size_t>(batch_count) * max_frames * feat_dim, 0.0f);
      for (int32_t bi = 0; bi != batch_count; ++bi) {
        const EmbeddingBatchItem &it = items[begin + bi];
        std::copy(it.features.begin(), it.features.end(),
                  buf.data() + static_cast<size_t>(bi) * max_frames * feat_dim);
        // Remaining (max_frames - it.num_frames) * feat_dim entries for
        // this item stay zero — the padding kEmbeddingLengthBucketMaxRatio
        // bounds.
      }

      auto memory_info =
          Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

      std::array<int64_t, 3> x_shape{batch_count, max_frames, feat_dim};
      Ort::Value x = Ort::Value::CreateTensor(memory_info, buf.data(),
                                              buf.size(), x_shape.data(),
                                              x_shape.size());

      Ort::Value embedding = model_.Compute(std::move(x));
      std::vector<int64_t> embedding_shape =
          embedding.GetTensorTypeAndShapeInfo().GetShape();
      // embedding_shape = {batch_count, embedding_dim}
      const int32_t embedding_dim = static_cast<int32_t>(embedding_shape[1]);
      const float *out_data = embedding.GetTensorData<float>();

      for (int32_t bi = 0; bi != batch_count; ++bi) {
        const EmbeddingBatchItem &it = items[begin + bi];
        std::vector<float> emb(embedding_dim);
        std::copy(out_data + static_cast<size_t>(bi) * embedding_dim,
                  out_data + static_cast<size_t>(bi + 1) * embedding_dim,
                  emb.begin());
        (*ans)[it.index] = std::move(emb);
      }
    } catch (const std::exception &e) {
      if (batch_count > 1) {
        const size_t mid = begin + (end - begin) / 2;
        ProcessEmbeddingBatch(items, begin, mid, feat_dim, ans);
        ProcessEmbeddingBatch(items, mid, end, feat_dim, ans);
        return;
      }

      // A single item alone still doesn't fit — genuinely unrecoverable
      // for THIS segment specifically (not a batching artifact). Marked
      // NaN (using the model's static output_dim, since a failed call
      // never produced a real result to read the dimension from) so the
      // EXISTING "embedding model may output NaN" handling in
      // ComputeEmbeddings() correctly filters it out via IsNaNWrapper,
      // instead of leaving an ambiguous empty vector that
      // std::none_of(empty range) would vacuously — and incorrectly —
      // treat as "valid, no NaN present".
      SHERPA_ONNX_LOGE(
          "Failed to compute embedding for one segment even at batch size "
          "1 (%d frames) — skipping just this segment: %s",
          items[begin].num_frames, e.what());
      (*ans)[items[begin].index] = std::vector<float>(
          model_.GetMetaData().output_dim, std::numeric_limits<float>::quiet_NaN());
    }
  }

 private:
  SpeakerEmbeddingExtractorModel model_;

  // suopTranscriber local patch: see ComputeBatch()'s doc comment for both.
  // Not tuned per-GPU. 32 mirrors the segmentation model's batch size (same
  // size/memory-triviality reasoning). 1.5x bounds worst-case padding
  // waste (and therefore worst-case pooled-embedding drift from padding)
  // for the shortest item in any one bucket, while still giving same-ish-
  // length segments (the common case for turns from the same speaker) room
  // to batch together.
  static constexpr int32_t kEmbeddingBatchSize = 32;
  static constexpr float kEmbeddingLengthBucketMaxRatio = 1.5f;

  // suopTranscriber local patch: see the bucketing loop's doc comment
  // above. A SOFT, best-effort cap on batch_count * max_frames — the
  // actual driver of a batch's GPU tensor size — meant to avoid the cost
  // of an allocation attempt we can predict is probably too large, NOT a
  // guarantee of safety. It can't be one: confirmed directly, an earlier
  // value of 100,000 here (raw input tensor ~32 MB) still let through a
  // batch whose real GPU memory need (internal activation memory for this
  // model, an unknown multiple of input size we have no visibility into
  // from here) hit a 1.1 GB single-allocation CUDA failure. The REAL
  // safety net is ProcessEmbeddingBatch()'s own catch-and-bisect fallback
  // — this constant only shifts how OFTEN that fallback has to run.
  // Lowered 5x (to 20,000, ~6.4 MB raw input tensor) after that failure,
  // still not derived from a measured safe maximum.
  static constexpr int64_t kEmbeddingMaxBatchFrames = 20000;

  // suopTranscriber local patch: see ProcessEmbeddingBatch()'s doc comment
  // on max_frames. 100 frames (~1s at the typical 100fps fbank rate) is a
  // coarse-but-not-wasteful rounding granularity — small enough that the
  // extra zero-padding it adds is minor next to what
  // kEmbeddingLengthBucketMaxRatio already tolerates, coarse enough to
  // meaningfully cut down how many DISTINCT tensor shapes a long
  // recording's many buckets collectively need, which is what actually
  // determines how large the CUDA memory arena grows to. Not tuned to a
  // measured optimum.
  static constexpr int32_t kEmbeddingFrameQuantum = 100;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_GENERAL_IMPL_H_
