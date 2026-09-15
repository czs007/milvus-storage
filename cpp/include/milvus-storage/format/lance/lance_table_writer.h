// Copyright 2025 Zilliz
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef BUILD_GTEST

#include <memory>
#include <string>
#include <vector>

#include <arrow/record_batch.h>
#include <arrow/result.h>
#include <arrow/status.h>

#include "milvus-storage/common/writer_status.h"
#include "milvus-storage/format/format_writer.h"
#include "milvus-storage/properties.h"
#include "lance/lance_bridge.h"

namespace milvus_storage::lance {

/**
 * Current writer won't used, except test
 */
class LanceTableWriter final : public FormatWriter {
  public:
  LanceTableWriter(const std::string& base_path,
                   std::shared_ptr<arrow::Schema> schema,
                   const api::Properties& properties,
                   LanceDataStorageFormat data_storage_format = LanceDataStorageFormat::Stable);

  ~LanceTableWriter() = default;

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch> record) override;

  arrow::Status Flush() override;

  arrow::Result<api::ColumnGroupFile> Close() override;

  void Abort() noexcept override;

  private:
  arrow::Status WriteImpl(const std::shared_ptr<arrow::RecordBatch>& record);
  arrow::Status FlushImpl();
  arrow::Result<api::ColumnGroupFile> CloseImpl();

  bool closed_;
  std::string base_path_;
  std::shared_ptr<arrow::Schema> schema_;
  api::Properties properties_;
  LanceDataStorageFormat data_storage_format_;

  std::vector<std::shared_ptr<arrow::RecordBatch>> record_batches_;
  WriterStatus writer_status_;
  int64_t written_rows_ = 0;
};
}  // namespace milvus_storage::lance

#endif  // BUILD_GTEST
