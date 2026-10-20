#include "artifact/schema.h"

#include <algorithm>
#include <limits>
#include <set>

namespace ninfer::artifact::v3 {
namespace {

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

// Closed member sets: every required member present, no unknown member, no JSON null where a
// concrete value is required.
void require_members(const Json& value, std::initializer_list<std::string_view> required,
                     std::initializer_list<std::string_view> optional, std::string_view label) {
    if (!value.is_object()) { throw ArtifactError(std::string(label) + " must be a JSON object"); }
    for (const auto& member : value.items()) {
        const std::string_view key = member.key();
        const bool known = std::find(required.begin(), required.end(), key) != required.end() ||
                           std::find(optional.begin(), optional.end(), key) != optional.end();
        if (!known) {
            throw ArtifactError(std::string(label) + " has unexpected member '" + std::string(key) +
                                "'");
        }
    }
    for (const std::string_view key : required) {
        if (!value.contains(key)) {
            throw ArtifactError(std::string(label) + " is missing member '" + std::string(key) +
                                "'");
        }
    }
}

const Json& require_member(const Json& value, std::string_view key, std::string_view label) {
    const auto it = value.find(key);
    if (it == value.end()) {
        throw ArtifactError(std::string(label) + " is missing member '" + std::string(key) + "'");
    }
    return *it;
}

const std::string& require_string(const Json& value, std::string_view label, bool nonempty = true) {
    if (!value.is_string()) { throw ArtifactError(std::string(label) + " must be a string"); }
    const auto& text = value.get_ref<const std::string&>();
    if (nonempty && text.empty()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    for (const char character : text) {
        if (character == '\0') {
            throw ArtifactError(std::string(label) + " must not contain U+0000");
        }
    }
    return text;
}

std::uint64_t require_u64(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be a nonnegative integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

Shape parse_shape(const Json& value, std::string_view label) {
    if (!value.is_array()) { throw ArtifactError(std::string(label) + " must be an array"); }
    if (value.size() > 16) {
        throw ArtifactError(std::string(label) + " must hold at most 16 dimensions");
    }
    Shape shape;
    shape.reserve(value.size());
    for (const auto& dimension : value) { shape.push_back(require_u64(dimension, label, true)); }
    return shape;
}

std::uint64_t element_count(const Shape& shape) {
    std::uint64_t elements = 1;
    for (const auto dimension : shape) {
        elements = checked_mul(elements, dimension, "shape element count");
    }
    return elements;
}

TensorObject parse_tensor(const Json& value) {
    require_members(value, {"id", "kind", "shape", "format", "layout", "offset", "bytes"}, {},
                    "tensor object");
    if (require_string(require_member(value, "kind", "tensor object"), "tensor kind") != "tensor") {
        throw ArtifactError("tensor kind must be 'tensor'");
    }
    TensorObject out;
    out.id     = require_string(require_member(value, "id", "tensor object"), "tensor id");
    out.shape  = parse_shape(require_member(value, "shape", "tensor object"), "tensor shape");
    out.format = require_string(require_member(value, "format", "tensor object"), "tensor format");
    out.layout = require_string(require_member(value, "layout", "tensor object"), "tensor layout");
    out.offset =
        require_u64(require_member(value, "offset", "tensor object"), "tensor offset", false);
    out.bytes = require_u64(require_member(value, "bytes", "tensor object"), "tensor bytes", true);
    return out;
}

ResourceObject parse_resource(const Json& value) {
    require_members(value, {"id", "kind", "encoding", "offset", "bytes"}, {}, "resource object");
    if (require_string(require_member(value, "kind", "resource object"), "resource kind") !=
        "resource") {
        throw ArtifactError("resource kind must be 'resource'");
    }
    ResourceObject out;
    out.id = require_string(require_member(value, "id", "resource object"), "resource id");
    out.encoding =
        require_string(require_member(value, "encoding", "resource object"), "resource encoding");
    out.offset =
        require_u64(require_member(value, "offset", "resource object"), "resource offset", false);
    out.bytes =
        require_u64(require_member(value, "bytes", "resource object"), "resource bytes", true);
    return out;
}

Object parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto kind = value.find("kind");
    if (kind == value.end()) { throw ArtifactError("object entry is missing member 'kind'"); }
    const auto& name = require_string(*kind, "object kind");
    if (name == "tensor") { return parse_tensor(value); }
    if (name == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

Proposal parse_proposal(const Json& value) {
    require_members(value, {"domain"}, {"rows"}, "proposal record");
    const auto& domain =
        require_string(require_member(value, "domain", "proposal domain"), "proposal domain");
    Proposal parsed;
    if (domain == "indexed") {
        parsed.indexed = true;
        parsed.rows =
            require_u64(require_member(value, "rows", "proposal rows"), "proposal rows", true);
    } else if (domain == "full") {
        if (value.contains("rows")) {
            throw ArtifactError("proposal rows are valid only for the 'indexed' domain");
        }
    } else {
        throw ArtifactError("proposal domain must be 'full' or 'indexed'");
    }
    return parsed;
}

Component parse_component(const Json& value, std::string_view id) {
    const std::string label = "component '" + std::string(id) + "'";
    require_members(value, {"config"}, {"target", "resources", "proposal"}, label);
    Component out;
    out.config = require_member(value, "config", label);
    if (!out.config.is_object()) { throw ArtifactError(label + " config must be a JSON object"); }
    if (const auto target = value.find("target"); target != value.end()) {
        out.target = require_string(*target, "component target");
    }
    if (const auto resources = value.find("resources"); resources != value.end()) {
        if (!resources->is_object()) {
            throw ArtifactError(label + " resources must be an object");
        }
        // Roles are resolved to object indices once the whole object table is known.
        for (const auto& item : resources->items()) {
            (void)require_string(item.value(), "component resource reference");
        }
    }
    if (const auto proposal = value.find("proposal"); proposal != value.end()) {
        out.proposal = parse_proposal(*proposal);
    }
    return out;
}

ActivationPolicy parse_policy(const Json& value) {
    const auto& name = require_string(value, "activation_policy");
    if (name == "A16Only") { return ActivationPolicy::A16Only; }
    if (name == "AllowA8") { return ActivationPolicy::AllowA8; }
    if (name == "AllowA4") { return ActivationPolicy::AllowA4; }
    throw ArtifactError("unknown activation_policy '" + name + "'");
}

FileRecord parse_file(const Json& value, bool entry) {
    require_members(value, {"path", "payload_bytes"}, {}, "files entry");
    FileRecord out;
    out.entry        = entry;
    const auto& path = require_member(value, "path", "files entry");
    if (entry) {
        if (!path.is_null()) { throw ArtifactError("files[0].path must be null"); }
    } else {
        out.path = require_string(path, "files path");
        if (out.path == "." || out.path == "..") {
            throw ArtifactError("files path must not be '.' or '..'");
        }
        if (out.path.find('/') != std::string::npos || out.path.find('\\') != std::string::npos) {
            throw ArtifactError("files path '" + out.path + "' must be a bare file name");
        }
    }
    out.payload_bytes = require_u64(require_member(value, "payload_bytes", "files entry"),
                                    "files payload_bytes", true);
    return out;
}

// Validates one Binding against the parsed object table and records its logical element count.
Binding parse_binding(const Json& value, const Directory& directory, std::string_view label) {
    if (!value.is_object()) { throw ArtifactError(std::string(label) + " must be a JSON object"); }
    Binding out;
    if (value.contains("object")) {
        require_members(value, {"object"}, {}, label);
        const auto& id = require_string(require_member(value, "object", label), "binding object");
        const Object* object = directory.find(id);
        if (object == nullptr) {
            throw ArtifactError(std::string(label) + " references unknown object '" + id + "'");
        }
        const auto* tensor = tensor_object(*object);
        if (tensor == nullptr) {
            throw ArtifactError(std::string(label) + " must reference a tensor object, got '" + id +
                                "'");
        }
        out.whole_object = true;
        out.object       = directory.object_index.at(id);
        out.elements     = element_count(tensor->shape);
        return out;
    }
    require_members(value, {"parts"}, {}, label);
    const auto& parts = require_member(value, "parts", label);
    if (!parts.is_array() || parts.empty()) {
        throw ArtifactError(std::string(label) + " parts must be a nonempty array");
    }
    out.elements = 0;
    out.parts.reserve(parts.size());
    for (const auto& entry : parts) {
        require_members(entry, {"object", "range"}, {}, "part");
        const auto& id = require_string(require_member(entry, "object", "part"), "part object");
        const Object* object = directory.find(id);
        if (object == nullptr) {
            throw ArtifactError(std::string(label) + " part references unknown object '" + id +
                                "'");
        }
        const auto* tensor = tensor_object(*object);
        if (tensor == nullptr) {
            throw ArtifactError(std::string(label) + " part must reference a tensor object, got '" +
                                id + "'");
        }
        const auto& range = require_member(entry, "range", "part");
        if (!range.is_array() || range.size() != 2) {
            throw ArtifactError("part range must be a two element array");
        }
        const auto begin = require_u64(range.at(0), "part range begin", false);
        const auto end   = require_u64(range.at(1), "part range end", false);
        const auto total = element_count(tensor->shape);
        if (begin >= end || end > total) {
            throw ArtifactError(std::string(label) + " part range [" + std::to_string(begin) + "," +
                                std::to_string(end) + ") is outside '" + id + "' elements " +
                                std::to_string(total));
        }
        out.elements = checked_add(out.elements, end - begin, "binding element coverage");
        out.parts.push_back({directory.object_index.at(id), begin, end});
    }
    return out;
}

void parse_use(const Json& value, Directory& directory) {
    require_members(value, {"parameter", "input"}, {"activation_policy", "auxiliaries"}, "use");
    Use out;
    out.parameter = require_string(require_member(value, "parameter", "use"), "use parameter");
    out.input     = require_string(require_member(value, "input", "use"), "use input");
    if (directory.bindings.find(out.parameter) == directory.bindings.end()) {
        throw ArtifactError("use references unknown parameter '" + out.parameter + "'");
    }
    const std::pair<std::string, std::string> key{out.parameter, out.input};
    if (directory.use_index.find(key) != directory.use_index.end()) {
        throw ArtifactError("duplicate use for parameter '" + out.parameter + "' input '" +
                            out.input + "'");
    }
    if (const auto policy = value.find("activation_policy"); policy != value.end()) {
        out.activation_policy = parse_policy(*policy);
    }
    if (const auto auxiliaries = value.find("auxiliaries"); auxiliaries != value.end()) {
        if (!auxiliaries->is_object()) { throw ArtifactError("use auxiliaries must be an object"); }
        for (const auto& item : auxiliaries->items()) {
            out.auxiliaries.emplace(item.key(), parse_binding(item.value(), directory,
                                                              "auxiliary '" + item.key() + "'"));
        }
    }
    directory.use_index.emplace(key, directory.uses.size());
    directory.uses.push_back(std::move(out));
}

} // namespace

const std::string& object_id(const Object& object) noexcept {
    return std::visit([](const auto& value) -> const std::string& { return value.id; }, object);
}

std::uint64_t object_offset(const Object& object) noexcept {
    return std::visit([](const auto& value) { return value.offset; }, object);
}

std::uint64_t object_bytes(const Object& object) noexcept {
    return std::visit([](const auto& value) { return value.bytes; }, object);
}

bool object_is_tensor(const Object& object) noexcept {
    return std::holds_alternative<TensorObject>(object);
}

const TensorObject* tensor_object(const Object& object) noexcept {
    return std::get_if<TensorObject>(&object);
}

const Object& Directory::object(std::size_t index) const {
    if (index >= objects.size()) { throw ArtifactError("object index is out of range"); }
    return objects[index];
}

const Object* Directory::find(std::string_view id) const {
    const auto it = object_index.find(std::string(id));
    return it == object_index.end() ? nullptr : &objects[it->second];
}

const Component& Directory::component(std::string_view id) const {
    const auto it = components.find(std::string(id));
    if (it == components.end()) {
        throw ArtifactError("artifact has no component '" + std::string(id) + "'");
    }
    return it->second;
}

const Use* Directory::use(std::string_view parameter, std::string_view input) const {
    const auto it = use_index.find({std::string(parameter), std::string(input)});
    return it == use_index.end() ? nullptr : &uses[it->second];
}

Directory parse_directory(const Json& root, std::string_view entry_name) {
    require_members(root, {"components", "objects", "bindings", "uses", "files"},
                    {"metadata", "provenance"}, "directory root");
    Directory directory;

    // Physical objects first: bindings, uses and component resource maps all reference them.
    const auto& raw_objects = require_member(root, "objects", "directory root");
    if (!raw_objects.is_array() || raw_objects.empty()) {
        throw ArtifactError("objects must be a nonempty array");
    }
    directory.objects.reserve(raw_objects.size());
    directory.object_index.reserve(raw_objects.size());
    for (const auto& raw_object : raw_objects) {
        Object object            = parse_object(raw_object);
        const auto& id           = object_id(object);
        const auto [_, inserted] = directory.object_index.emplace(id, directory.objects.size());
        if (!inserted) { throw ArtifactError("duplicate object id '" + id + "'"); }
        directory.objects.push_back(std::move(object));
    }

    const auto& raw_bindings = require_member(root, "bindings", "directory root");
    if (!raw_bindings.is_object()) { throw ArtifactError("bindings must be a JSON object"); }
    for (const auto& item : raw_bindings.items()) {
        directory.bindings.emplace(
            item.key(), parse_binding(item.value(), directory, "binding '" + item.key() + "'"));
    }
    const auto& raw_uses = require_member(root, "uses", "directory root");
    if (!raw_uses.is_array()) { throw ArtifactError("uses must be an array"); }
    for (const auto& raw_use : raw_uses) { parse_use(raw_use, directory); }

    // Component records, including their resource and target references.
    const auto& raw_components = require_member(root, "components", "directory root");
    if (!raw_components.is_object()) { throw ArtifactError("components must be a JSON object"); }
    for (const auto& item : raw_components.items()) {
        directory.components.emplace(item.key(), parse_component(item.value(), item.key()));
    }
    if (directory.components.find("text") == directory.components.end()) {
        throw ArtifactError("components must contain 'text'");
    }
    for (const auto& [id, component] : directory.components) {
        if (component.target.has_value() &&
            directory.components.find(*component.target) == directory.components.end()) {
            throw ArtifactError("component '" + id + "' targets unknown component '" +
                                *component.target + "'");
        }
        const auto raw = raw_components.find(id);
        if (raw == raw_components.end()) { continue; }
        const auto roles = raw->find("resources");
        if (roles == raw->end()) { continue; }
        for (const auto& role : roles->items()) {
            const auto& reference = require_string(role.value(), "component resource reference");
            const Object* object  = directory.find(reference);
            if (object == nullptr) {
                throw ArtifactError("component '" + id + "' resource '" + role.key() +
                                    "' references unknown object '" + reference + "'");
            }
            if (object_is_tensor(*object)) {
                throw ArtifactError("component '" + id + "' resource '" + role.key() +
                                    "' must reference a resource object");
            }
            directory.components.at(id).resources[role.key()] =
                directory.object_index.at(reference);
        }
    }

    // File directory and the logical payload prefix sums.
    const auto& raw_files = require_member(root, "files", "directory root");
    if (!raw_files.is_array() || raw_files.empty()) {
        throw ArtifactError("files must be a nonempty array");
    }
    std::set<std::string> part_paths;
    std::uint64_t cursor = 0;
    for (std::size_t index = 0; index < raw_files.size(); ++index) {
        FileRecord record    = parse_file(raw_files.at(index), index == 0);
        record.logical_begin = cursor;
        cursor               = checked_add(cursor, record.payload_bytes, "logical payload");
        if (!record.entry) {
            if (record.path == std::string(entry_name)) {
                throw ArtifactError("continuation volume '" + record.path +
                                    "' collides with the entry file name");
            }
            if (!part_paths.insert(record.path).second) {
                throw ArtifactError("duplicate continuation volume name '" + record.path + "'");
            }
        }
        directory.files.push_back(std::move(record));
    }
    directory.payload_bytes = cursor;

    if (const auto metadata = root.find("metadata"); metadata != root.end()) {
        if (!metadata->is_object()) { throw ArtifactError("metadata must be a JSON object"); }
        directory.metadata = *metadata;
        if (const auto name = metadata->find("name"); name != metadata->end()) {
            directory.name = require_string(*name, "metadata name");
        }
    }
    if (const auto provenance = root.find("provenance"); provenance != root.end()) {
        if (!provenance->is_object()) { throw ArtifactError("provenance must be a JSON object"); }
        directory.provenance = *provenance;
    }

    return directory;
}

} // namespace ninfer::artifact::v3
