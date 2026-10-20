#include "artifact/binder.h"

#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

std::uint64_t element_count(std::span<const std::uint64_t> shape) {
    std::uint64_t elements = 1;
    for (const auto dimension : shape) {
        if (dimension != 0 && elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw ArtifactError("shape element count overflows u64");
        }
        elements *= dimension;
    }
    return elements;
}

// One parent object plus the logical element ranges that address it.
struct ResolvedRegion {
    std::size_t object = 0;
    bool assigned      = false;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
};

void merge_region(ResolvedRegion& region, std::size_t object, std::uint64_t begin,
                  std::uint64_t end, std::string_view label) {
    if (region.assigned && region.object != object) {
        throw ArtifactError(std::string(label) + " spans several stored objects (index " +
                            std::to_string(region.object) + " and " + std::to_string(object) + ")");
    }
    region.object   = object;
    region.assigned = true;
    region.ranges.emplace_back(begin, end);
}

ResolvedRegion resolve_binding(const v3::Directory& directory, const v3::Binding& binding,
                               std::string_view label) {
    ResolvedRegion region;
    if (binding.whole_object) {
        const auto& tensor = *v3::tensor_object(directory.object(binding.object));
        merge_region(region, binding.object, 0, element_count(tensor.shape), label);
        return region;
    }
    for (const auto& part : binding.parts) {
        merge_region(region, part.object, part.begin, part.end, label);
    }
    return region;
}

ResolvedRegion resolve_group(const v3::Directory& directory,
                             std::span<const std::string_view> names) {
    ResolvedRegion region;
    for (const std::string_view name : names) {
        const auto it = directory.bindings.find(name);
        if (it == directory.bindings.end()) {
            throw ArtifactError("logical parameter is missing: " + std::string(name));
        }
        const ResolvedRegion one = resolve_binding(directory, it->second, name);
        for (const auto& [begin, end] : one.ranges) {
            merge_region(region, one.object, begin, end, name);
        }
    }
    return region;
}

// The ranges must tile the parent exactly once from element zero. Repeating one identical range is
// legal: a training parameter can be consumed at two use positions (DFlash2 context K/V alias the
// draft K/V window), and each part then names the same region.
void require_full_cover(const ResolvedRegion& region, std::uint64_t elements,
                        std::string_view label) {
    auto ranges = region.ranges;
    std::sort(ranges.begin(), ranges.end());
    ranges.erase(std::unique(ranges.begin(), ranges.end()), ranges.end());
    std::uint64_t cursor  = 0;
    std::uint64_t covered = 0;
    bool contiguous       = true;
    for (const auto& [begin, end] : ranges) {
        if (begin != cursor) { contiguous = false; }
        cursor = end;
        covered += end - begin;
    }
    if (contiguous && cursor == elements) { return; }
    throw ArtifactError(std::string(label) + " covers " + std::to_string(covered) + " of " +
                        std::to_string(elements) + " elements of one parent object" +
                        (contiguous ? "" : " and leaves a gap") +
                        "; a parameter that is only a range of a fused parent must be bound "
                        "through require_group");
}

// A logical parameter that owns its object, or a fusion group, must address the whole parent.
void require_whole_parent(const Binder& binder, const ResolvedRegion& region,
                          std::string_view label) {
    const ObjectHandle handle{region.object};
    const auto* tensor = std::get_if<TensorDescriptor>(&binder.descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(label));
    }
    require_full_cover(region, element_count(tensor->shape), label);
}

std::string join_names(std::span<const std::string_view> names) {
    std::string out;
    for (const std::string_view name : names) {
        if (!out.empty()) { out += " + "; }
        out += name;
    }
    return out;
}

} // namespace

Binder::Binder(const Reader& reader)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false) {
    materialization_.object_count = reader.objects().size();
}

std::uint32_t Binder::version() const noexcept { return reader_.version(); }

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects            = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

// A v3 directory index and the flattened descriptor list must stay aligned; the reader builds both
// from the same ordered object array.
ObjectHandle Binder::use_object(std::size_t index, std::string_view label) {
    if (index >= consumed_.size()) {
        throw ArtifactError(std::string(label) + " resolves to an object outside the directory");
    }
    if (reader_.version() == 3) {
        const auto& object = reader_.directory().object(index);
        if (object_name(reader_.objects()[index]) != v3::object_id(object)) {
            throw ArtifactError("v3 object table and descriptor list disagree at index " +
                                std::to_string(index));
        }
    }
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(label));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

void Binder::require_tensor_contract(ObjectHandle handle, NumericFormat format,
                                     StorageLayout layout, std::span<const std::uint64_t> shape,
                                     std::string_view label) const {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(label));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(label));
    }
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout, std::span<const std::uint64_t> shape) {
    if (reader_.version() != 3) {
        const ObjectHandle handle = find_unconsumed(name);
        require_tensor_contract(handle, format, layout, shape, name);
        return handle;
    }

    const auto& directory = reader_.directory();
    const auto it         = directory.bindings.find(name);
    if (it == directory.bindings.end()) {
        throw ArtifactError("logical parameter is missing: " + std::string(name));
    }
    const ResolvedRegion region = resolve_binding(directory, it->second, name);
    const ObjectHandle handle{region.object};
    require_whole_parent(*this, region, name);
    require_tensor_contract(handle, format, layout, shape, name);
    return use_object(region.object, name);
}

ObjectHandle Binder::require_group(std::span<const std::string_view> names, NumericFormat format,
                                   StorageLayout layout, std::span<const std::uint64_t> shape) {
    if (names.empty()) { throw ArtifactError("a fusion group needs at least one parameter"); }
    if (reader_.version() != 3) {
        throw ArtifactError("a fusion group is a v3 logical-parameter lookup");
    }
    const std::string label     = join_names(names);
    const ResolvedRegion region = resolve_group(reader_.directory(), names);
    const ObjectHandle handle{region.object};
    require_whole_parent(*this, region, label);
    require_tensor_contract(handle, format, layout, shape, label);
    return use_object(region.object, label);
}

ObjectHandle Binder::require_auxiliary(std::string_view parameter, std::string_view input,
                                       std::string_view role, NumericFormat format,
                                       StorageLayout layout, std::span<const std::uint64_t> shape) {
    if (reader_.version() != 3) { throw ArtifactError("a use auxiliary is a v3 lookup"); }
    const std::string label =
        std::string(parameter) + " @" + std::string(input) + " / " + std::string(role);
    const v3::Use* use = reader_.directory().use(parameter, input);
    if (use == nullptr) {
        throw ArtifactError("artifact has no use for parameter '" + std::string(parameter) +
                            "' input '" + std::string(input) + "'");
    }
    const auto it = use->auxiliaries.find(role);
    if (it == use->auxiliaries.end()) {
        throw ArtifactError("use '" + label + "' has no auxiliary");
    }
    const ResolvedRegion region = resolve_binding(reader_.directory(), it->second, label);
    const ObjectHandle handle{region.object};
    require_whole_parent(*this, region, label);
    require_tensor_contract(handle, format, layout, shape, label);
    return use_object(region.object, label);
}

ObjectHandle Binder::require_component_resource(std::string_view component, std::string_view role,
                                                ResourceEncoding encoding) {
    if (reader_.version() != 3) { throw ArtifactError("a component resource role is a v3 lookup"); }
    const std::string label = std::string(component) + "/" + std::string(role);
    const auto& directory   = reader_.directory();
    const auto& record      = directory.component(component);
    const auto it           = record.resources.find(role);
    if (it == record.resources.end()) {
        throw ArtifactError("component '" + std::string(component) + "' declares no resource '" +
                            std::string(role) + "'");
    }
    const ObjectHandle handle = use_object(it->second, label);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) { throw ArtifactError("component resource is a tensor: " + label); }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " + label);
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

bool Binder::contains(std::string_view name) const noexcept {
    if (reader_.version() == 3) {
        const auto& bindings = reader_.directory().bindings;
        return bindings.find(name) != bindings.end();
    }
    return reader_.find(name) != nullptr;
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const std::uint64_t offset    = align_up(materialization_.device_capacity_bytes, alignment);
    if (tensor->bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(
        DeviceMaterialization{handle, offset, tensor->bytes, alignment});
    materialization_.device_capacity_bytes = offset + tensor->bytes;
    planned_[handle.index]                 = true;
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(resource->name));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::validate_only(ObjectHandle handle) {
    const ObjectDescriptor& object = descriptor(handle);
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(object)));
    }
    planned_[handle.index] = true;
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    return std::move(materialization_);
}

} // namespace ninfer::artifact
