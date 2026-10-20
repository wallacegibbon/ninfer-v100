#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::artifact {

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
};

struct ObjectHandle {
    std::size_t index = 0;
};

struct DeviceMaterialization {
    ObjectHandle object;
    std::uint64_t offset    = 0;
    std::uint64_t bytes     = 0;
    std::uint64_t alignment = 0;
};

struct HostMaterialization {
    ObjectHandle object;
};

struct MaterializationPlan {
    std::size_t object_count            = 0;
    std::uint64_t device_capacity_bytes = 0;
    std::vector<DeviceMaterialization> device_objects;
    std::vector<HostMaterialization> host_objects;
};

class Binder {
public:
    explicit Binder(const Reader& reader);

    ObjectHandle require_tensor(std::string_view name, NumericFormat format, StorageLayout layout,
                                std::span<const std::uint64_t> shape);
    ObjectHandle require_resource(std::string_view name, ResourceEncoding encoding);

    // Version-3 lookups. A v3 stored object is addressed by an opaque id, so the target asks for
    // the logical parameters it consumes instead. `require_group` resolves the set of parameters a
    // native op consumes as one fused weight and requires them to tile exactly one parent object,
    // which is the shape those ops already take. `require_auxiliary` resolves a per-use auxiliary
    // such as the activation input divisor. `require_component_resource` resolves a frontend
    // resource through the component's role map.
    ObjectHandle require_group(std::span<const std::string_view> names, NumericFormat format,
                               StorageLayout layout, std::span<const std::uint64_t> shape);
    ObjectHandle require_auxiliary(std::string_view parameter, std::string_view input,
                                   std::string_view role, NumericFormat format,
                                   StorageLayout layout, std::span<const std::uint64_t> shape);
    ObjectHandle require_component_resource(std::string_view component, std::string_view role,
                                            ResourceEncoding encoding);

    [[nodiscard]] std::uint32_t version() const noexcept;

    [[nodiscard]] const Reader& reader() const noexcept { return reader_; }

    [[nodiscard]] bool contains(std::string_view name) const noexcept;
    const ObjectDescriptor& descriptor(ObjectHandle handle) const;
    PayloadSpan payload(ObjectHandle handle) const;
    void materialize_on_device(ObjectHandle handle);
    void retain_on_host(ObjectHandle handle);
    void validate_only(ObjectHandle handle);
    MaterializationPlan finish();

private:
    ObjectHandle find_unconsumed(std::string_view name);
    ObjectHandle use_object(std::size_t index, std::string_view label);
    void require_tensor_contract(ObjectHandle handle, NumericFormat format, StorageLayout layout,
                                 std::span<const std::uint64_t> shape,
                                 std::string_view label) const;

    const Reader& reader_;
    std::vector<bool> consumed_;
    std::vector<bool> planned_;
    MaterializationPlan materialization_;
};

} // namespace ninfer::artifact
