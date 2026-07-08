// sherpa-onnx/csrc/speaker-embedding-extractor-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_
#define SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/speaker-embedding-extractor.h"

namespace sherpa_onnx {

class SpeakerEmbeddingExtractorImpl {
 public:
  virtual ~SpeakerEmbeddingExtractorImpl() = default;

  static std::unique_ptr<SpeakerEmbeddingExtractorImpl> Create(
      const SpeakerEmbeddingExtractorConfig &config);

  template <typename Manager>
  static std::unique_ptr<SpeakerEmbeddingExtractorImpl> Create(
      Manager *mgr, const SpeakerEmbeddingExtractorConfig &config);

  virtual int32_t Dim() const = 0;

  virtual std::unique_ptr<OnlineStream> CreateStream() const = 0;

  virtual bool IsReady(OnlineStream *s) const = 0;

  virtual std::vector<float> Compute(OnlineStream *s) const = 0;

  // suopTranscriber local patch (see patches/sherpa-onnx-batch-
  // segmentation.patch — same patch file covers both the segmentation and
  // embedding batching changes): batched form of Compute() above.
  // progressCallback (optional), if given, is invoked with (numProcessed,
  // numTotal) — the granularity is "once per completed batch/bucket" for
  // real batched impls, or once per item for this naive default (called
  // right after each Compute(), same cadence the OLD non-batched call site
  // used to report at) — so callers get roughly the same progress
  // responsiveness they always did, regardless of whether the underlying
  // impl actually batches. Default implementation is a naive per-stream
  // loop — numerically IDENTICAL to calling Compute() on each stream
  // individually, so any impl that doesn't override this (e.g. NeMo) keeps
  // today's exact behavior for free. Overridden in
  // SpeakerEmbeddingExtractorGeneralImpl (the impl WeSpeaker/CAM++ etc.
  // actually use) with a real batched fast path.
  virtual std::vector<std::vector<float>> ComputeBatch(
      const std::vector<OnlineStream *> &streams,
      const std::function<void(int32_t, int32_t)> &progressCallback = {}) const {
    std::vector<std::vector<float>> ans;
    ans.reserve(streams.size());
    const int32_t total = static_cast<int32_t>(streams.size());
    int32_t done = 0;
    for (OnlineStream *s : streams) {
      ans.push_back(Compute(s));
      ++done;
      if (progressCallback) {
        progressCallback(done, total);
      }
    }
    return ans;
  }
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_EMBEDDING_EXTRACTOR_IMPL_H_
