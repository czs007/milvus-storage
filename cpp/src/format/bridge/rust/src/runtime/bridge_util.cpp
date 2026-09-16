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

#include "runtime/bridge_util.h"

#include "runtime/bridge_error.h"

namespace milvus_storage {

// All three forms defer to the one decoder in bridge_error.cpp so the marker
// table cannot drift between the per-format bridges.
arrow::Status MakeBridgeErrorStatus(std::string_view message) { return bridge::MakeBridgeErrorStatus(message); }

arrow::Status MakeBridgeErrorStatus(std::string_view context, std::string_view message) {
  return bridge::WithBridgeContext(context, bridge::MakeBridgeErrorStatus(message));
}

arrow::Status MakeBridgeErrorStatus(std::string_view context, const arrow::Status& status) {
  return bridge::TranslateBridgeStatus(context, status);
}

}  // namespace milvus_storage
