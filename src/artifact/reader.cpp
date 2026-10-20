#include "artifact/reader.h"

#include "artifact/framing.h"
#include "artifact/schema.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    const auto format = find_format(name);
    if (!format.has_value()) { throw ArtifactError("unknown tensor format: " + std::string(name)); }
    return *format;
}

StorageLayout parse_layout(std::string_view name) {
    const auto layout = find_layout(name);
    if (!layout.has_value()) { throw ArtifactError("unknown tensor layout: " + std::string(name)); }
    return *layout;
}

ResourceEncoding parse_encoding(std::string_view name) {
    const auto encoding = find_encoding(name);
    if (!encoding.has_value()) {
        throw ArtifactError("unknown resource encoding: " + std::string(name));
    }
    return *encoding;
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
    }

    ~MappedFile() {
        if (data_ != nullptr) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
    }

private:
    int fd_                = -1;
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

template <std::size_t N>
bool magic_matches(std::span<const std::byte> data, const std::array<std::byte, N>& magic) {
    return data.size() >= N && std::equal(magic.begin(), magic.end(), data.begin());
}

} // namespace

std::optional<NumericFormat> find_format(std::string_view name) noexcept {
    // Version-2 spellings.
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    // Version-3 spellings.
    if (name == "bf16") { return NumericFormat::BF16; }
    if (name == "fp32") { return NumericFormat::FP32; }
    if (name == "int32") { return NumericFormat::I32; }
    if (name == "q4_g64_fp16") { return NumericFormat::Q4G64_F16S; }
    if (name == "q5_g64_fp16") { return NumericFormat::Q5G64_F16S; }
    if (name == "q6_g64_fp16") { return NumericFormat::Q6G64_F16S; }
    if (name == "q8_g32_fp16") { return NumericFormat::W8G32_F16S; }
    if (name == "nvfp4") { return NumericFormat::NVFP4; }
    if (name == "fp8_e4m3fn_row_bf16") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    return std::nullopt;
}

std::optional<StorageLayout> find_layout(std::string_view name) noexcept {
    if (name == "contiguous-le-v1" || name == "contiguous_le_v1") {
        return StorageLayout::ContiguousLeV1;
    }
    if (name == "row-split-k128-v1" || name == "row_split_k128_v1") {
        return StorageLayout::RowSplitK128V1;
    }
    if (name == "blockscale-k16-m128x4-v1" || name == "block_scale_k16_m128x4_v1") {
        return StorageLayout::BlockScaleK16M128x4V1;
    }
    if (name == "row-scale-v1" || name == "row_scale_v1") { return StorageLayout::RowScaleV1; }
    return std::nullopt;
}

std::optional<ResourceEncoding> find_encoding(std::string_view name) noexcept {
    if (name == "raw-bytes-v1" || name == "raw_bytes_v1") return ResourceEncoding::RawBytesV1;
    return std::nullopt;
}

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) {
        if (magic_matches(std::span(file.data(), file.size()), kEntryMagicV2)) {
            revision = 2;
            load_v2();
        } else if (magic_matches(std::span(file.data(), file.size()), kEntryMagicV3)) {
            revision = 3;
            load_v3(path);
        } else if (file.size() < kHeaderV2Bytes) {
            throw ArtifactError("artifact is shorter than the container header");
        } else {
            throw ArtifactError("artifact magic is not NInfer v2 or v3");
        }
    }

    // Version-2: sixteen-byte prefix, one flat directory, one payload region.
    void load_v2() {
        if (file.size() < kHeaderV2Bytes) {
            throw ArtifactError("artifact is shorter than the v2 prefix");
        }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kHeaderV2Bytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kHeaderV2Bytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);
            require_geometry(name, offset, bytes, alignment, payload_bytes, cursor);
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            cursor = offset + bytes;
        }
    }

    // Version-3: thirty-two-byte header, component records, logical bindings and uses, file
    // directory. Only a single-file set is interpreted for now; a continuation volume set is
    // rejected instead of silently reading a partial logical payload.
    void load_v3(const std::filesystem::path& path) {
        if (file.size() < kHeaderV3Bytes) {
            throw ArtifactError("artifact is shorter than the v3 header");
        }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(kHeaderV3Bytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }
        std::copy_n(file.data() + 16, kArtifactIdBytes, artifact_id.begin());

        Json root;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + kHeaderV3Bytes);
            root              = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        artifact_directory.emplace(v3::parse_directory(root, path.filename().string()));
        if (artifact_directory->files.size() != 1) {
            throw ArtifactError("v3 file sets with " +
                                std::to_string(artifact_directory->files.size()) +
                                " volumes are not supported yet; only the entry volume "
                                "can be read");
        }
        const auto expected = checked_add(
            payload_start, artifact_directory->files.front().payload_bytes, "entry file length");
        if (expected != file.size()) {
            throw ArtifactError("entry file is " + std::to_string(file.size()) +
                                " bytes; the files table declares " + std::to_string(expected));
        }

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        entries.reserve(artifact_directory->objects.size());
        index.reserve(artifact_directory->objects.size());
        std::uint64_t cursor = 0;
        for (const auto& object : artifact_directory->objects) {
            const auto& id    = v3::object_id(object);
            const auto offset = v3::object_offset(object);
            const auto bytes  = v3::object_bytes(object);
            if (const auto* tensor = v3::tensor_object(object)) {
                const auto format = find_format(tensor->format);
                if (!format.has_value()) {
                    throw ArtifactError("tensor '" + id + "' stores unregistered format '" +
                                        tensor->format + "'");
                }
                const auto layout = find_layout(tensor->layout);
                if (!layout.has_value()) {
                    throw ArtifactError("tensor '" + id + "' stores unregistered layout '" +
                                        tensor->layout + "'");
                }
                const auto expected_bytes = tensor_encoded_size(*layout, *format, tensor->shape);
                if (expected_bytes != bytes) {
                    throw ArtifactError("tensor '" + id + "' stores " + std::to_string(bytes) +
                                        " bytes; layout requires " +
                                        std::to_string(expected_bytes));
                }
                require_geometry(id, offset, bytes, tensor_alignment(*layout), payload_bytes,
                                 cursor);
                TensorDescriptor descriptor{id, tensor->shape, *format, *layout, offset, bytes};
                const auto object_index = entries.size();
                if (!index.emplace(id, object_index).second) {
                    throw ArtifactError("duplicate object id: " + id);
                }
                entries.push_back(std::move(descriptor));
            } else {
                const auto& resource = std::get<v3::ResourceObject>(object);
                const auto encoding  = find_encoding(resource.encoding);
                if (!encoding.has_value()) {
                    throw ArtifactError("resource '" + id + "' stores unregistered encoding '" +
                                        resource.encoding + "'");
                }
                require_geometry(id, offset, bytes, resource_alignment(*encoding), payload_bytes,
                                 cursor);
                ResourceDescriptor descriptor{id, *encoding, offset, bytes};
                const auto object_index = entries.size();
                if (!index.emplace(id, object_index).second) {
                    throw ArtifactError("duplicate object id: " + id);
                }
                entries.push_back(std::move(descriptor));
            }
            cursor = offset + bytes;
        }
    }

    static void require_geometry(std::string_view name, std::uint64_t offset, std::uint64_t bytes,
                                 std::uint64_t alignment, std::uint64_t payload_bytes,
                                 std::uint64_t cursor) {
        if (offset < cursor) {
            throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
        }
        if (alignment != 0 && offset % alignment != 0) {
            throw ArtifactError("object " + std::string(name) + " is not " +
                                std::to_string(alignment) + "-byte aligned");
        }
        const auto end = checked_add(offset, bytes, "object payload range");
        if (end > payload_bytes) {
            throw ArtifactError("object " + std::string(name) + " extends beyond the payload");
        }
    }

    MappedFile file;
    std::uint32_t revision = 2;
    ArtifactIdentity identity;
    std::array<std::byte, kArtifactIdBytes> artifact_id{};
    std::optional<v3::Directory> artifact_directory;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

std::uint32_t Reader::version() const noexcept { return impl_->revision; }

const ArtifactIdentity& Reader::identity() const {
    if (impl_->revision != 2) {
        throw ArtifactError(
            "a v3 artifact carries component records instead of a model identity; read "
            "directory().components");
    }
    return impl_->identity;
}

const v3::Directory& Reader::directory() const {
    if (!impl_->artifact_directory.has_value()) {
        throw ArtifactError("a v2 artifact has no v3 component directory");
    }
    return *impl_->artifact_directory;
}

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

std::span<const std::byte> Reader::artifact_id() const noexcept {
    if (impl_->revision != 3) { return {}; }
    return {impl_->artifact_id.data(), impl_->artifact_id.size()};
}

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
