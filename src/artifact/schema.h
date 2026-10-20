#pragma once

#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Version-3 container directory. The generic layer validates framing, the closed record sets,
// object geometry and reference integrity. It does not interpret architecture config, logical
// parameter shapes or format/layout semantics; those stay with the consumer.
namespace ninfer::artifact::v3 {

using Json  = nlohmann::json;
using Shape = std::vector<std::uint64_t>;

struct TensorObject {
    std::string id;
    Shape shape;
    std::string format; // stored spelling, resolved by the consumer
    std::string layout; // stored spelling, resolved by the consumer
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

struct ResourceObject {
    std::string id;
    std::string encoding;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

using Object = std::variant<TensorObject, ResourceObject>;

[[nodiscard]] const std::string& object_id(const Object& object) noexcept;
[[nodiscard]] std::uint64_t object_offset(const Object& object) noexcept;
[[nodiscard]] std::uint64_t object_bytes(const Object& object) noexcept;
[[nodiscard]] bool object_is_tensor(const Object& object) noexcept;
[[nodiscard]] const TensorObject* tensor_object(const Object& object) noexcept;

// Ordered logical elements of one stored object.
struct Part {
    std::uint64_t object = 0;
    std::uint64_t begin  = 0;
    std::uint64_t end    = 0;
};

struct Binding {
    bool whole_object    = false;
    std::uint64_t object = 0;   // valid for the whole-object form
    std::vector<Part> parts;    // valid for the ordered-parts form
    std::uint64_t elements = 0; // logical element count of the bound parameter
};

enum class ActivationPolicy {
    A16Only,
    AllowA8,
    AllowA4,
};

struct Use {
    std::string parameter;
    std::string input;
    std::optional<ActivationPolicy> activation_policy;
    std::map<std::string, Binding, std::less<>> auxiliaries;
};

struct Proposal {
    bool indexed       = false;
    std::uint64_t rows = 0;
};

struct Component {
    Json config;
    std::optional<std::string> target;
    std::map<std::string, std::size_t, std::less<>> resources; // role -> object index
    std::optional<Proposal> proposal;
};

struct FileRecord {
    bool entry = false;
    std::string path; // empty for the entry
    std::uint64_t payload_bytes = 0;
    std::uint64_t logical_begin = 0;
};

struct Directory {
    std::map<std::string, Component, std::less<>> components;
    std::vector<Object> objects;
    std::unordered_map<std::string, std::size_t> object_index;
    std::map<std::string, Binding, std::less<>> bindings;
    std::vector<Use> uses;
    std::map<std::pair<std::string, std::string>, std::size_t> use_index;
    std::vector<FileRecord> files;
    std::uint64_t payload_bytes = 0;
    Json metadata               = Json::object();
    Json provenance             = Json::object();
    std::string name; // metadata.name when present

    [[nodiscard]] const Object& object(std::size_t index) const;
    [[nodiscard]] const Object* find(std::string_view id) const;
    [[nodiscard]] const Component& component(std::string_view id) const;
    [[nodiscard]] const Use* use(std::string_view parameter, std::string_view input) const;
};

// Parses and validates one complete `NINFER\0\3` root record. `entry_name` is the opened entry
// file's base name, used to reject a continuation volume that collides with it.
[[nodiscard]] Directory parse_directory(const Json& root, std::string_view entry_name);

} // namespace ninfer::artifact::v3
