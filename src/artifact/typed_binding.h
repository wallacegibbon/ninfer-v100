#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace ninfer::artifact {

class MaterializedArtifact;

[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

// Version-3 fused weight: the logical parameters a native op consumes as one parent object.
[[nodiscard]] ObjectHandle bind_group(Binder& binder, std::span<const std::string_view> names,
                                      NumericFormat format,
                                      std::initializer_list<std::uint64_t> shape,
                                      TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_group(Binder& binder,
                                             std::span<const std::string_view> names,
                                             NumericFormat format,
                                             std::initializer_list<std::uint64_t> shape);

// Convenience: builds the ordered logical names of one fusion group from a block prefix and role
// suffixes ("attention/query", ...) and binds it in one call.
[[nodiscard]] ObjectHandle bind_group_roles(Binder& binder, std::string_view prefix,
                                            std::initializer_list<std::string_view> roles,
                                            NumericFormat format,
                                            std::initializer_list<std::uint64_t> shape,
                                            TensorPlacement placement);

// Version-3 per-use auxiliary value, for example the activation input divisor.
[[nodiscard]] ObjectHandle bind_auxiliary(Binder& binder, std::string_view parameter,
                                          std::string_view input, NumericFormat format,
                                          std::initializer_list<std::uint64_t> shape,
                                          TensorPlacement placement);

// Version-3 frontend resource selected through a component's role map.
[[nodiscard]] ObjectHandle bind_component_resource(Binder& binder, std::string_view component,
                                                   std::string_view role);

[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);

} // namespace ninfer::artifact
