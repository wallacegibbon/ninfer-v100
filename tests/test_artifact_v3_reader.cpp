// Generic v3 container reader test. Without an artifact to read it exits 77 (skipped), so the
// default test run stays hermetic. Pass a path as argv[1] or set NINFER_V3_ARTIFACT to run it
// against a real file:
//
//   ./build-v100/tests/ninfer_artifact_v3_reader_test ~/models/.../qwen3_8_27b_nvfp4.ninfer

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/schema.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kSkipCode = 77;

using ninfer::artifact::NumericFormat;
using ninfer::artifact::ObjectDescriptor;
using ninfer::artifact::Reader;
using ninfer::artifact::ResourceDescriptor;
using ninfer::artifact::StorageLayout;
using ninfer::artifact::TensorDescriptor;
namespace v3 = ninfer::artifact::v3;

void expect(bool condition, const std::string& what) {
    if (!condition) { throw std::runtime_error(what); }
}

std::string hex(std::span<const std::byte> bytes) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (const std::byte value : bytes) {
        const auto byte = std::to_integer<unsigned>(value);
        out.push_back(kDigits[byte >> 4U]);
        out.push_back(kDigits[byte & 0x0FU]);
    }
    return out;
}

std::uint64_t elements(std::span<const std::uint64_t> shape) {
    std::uint64_t count = 1;
    for (const auto dimension : shape) { count *= dimension; }
    return count;
}

// Counts parents whose logical ranges are referenced more than once. Reuse of one range by two
// names is legal (DFlash2 context K/V alias the draft K/V window), so the reader must allow it.
std::size_t count_aliased_parents(const v3::Directory& directory) {
    std::map<std::size_t, std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>> seen;
    for (const auto& [name, binding] : directory.bindings) {
        if (binding.whole_object) { continue; }
        for (const auto& part : binding.parts) { ++seen[part.object][{part.begin, part.end}]; }
    }
    std::size_t aliased = 0;
    for (const auto& [object, ranges] : seen) {
        (void)object;
        for (const auto& [range, count] : ranges) {
            (void)range;
            if (count > 1) {
                ++aliased;
                break;
            }
        }
    }
    return aliased;
}

// Sums the logical element coverage of a fusion group and requires it to be one contiguous prefix
// of a single parent, which is what the native ops consume as one fused weight.
struct GroupCoverage {
    std::size_t object  = 0;
    std::uint64_t first = 0;
    std::uint64_t last  = 0;
    std::uint64_t total = 0;
    bool contiguous     = false;
};

GroupCoverage group_coverage(const v3::Directory& directory,
                             std::initializer_list<std::string_view> names) {
    std::map<std::size_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> ranges;
    for (const std::string_view name : names) {
        const auto it = directory.bindings.find(name);
        expect(it != directory.bindings.end(), "missing binding '" + std::string(name) + "'");
        const auto& binding = it->second;
        if (binding.whole_object) {
            const auto& tensor = *v3::tensor_object(directory.object(binding.object));
            ranges[binding.object].emplace_back(0, elements(tensor.shape));
            continue;
        }
        for (const auto& part : binding.parts) {
            ranges[part.object].emplace_back(part.begin, part.end);
        }
    }
    expect(ranges.size() == 1, "fusion group spans several parent objects");
    GroupCoverage out;
    out.object   = ranges.begin()->first;
    auto entries = ranges.begin()->second;
    std::sort(entries.begin(), entries.end());
    out.contiguous       = entries.front().first == 0;
    std::uint64_t cursor = 0;
    for (const auto& [begin, end] : entries) {
        if (begin != cursor) { out.contiguous = false; }
        cursor += end - begin;
    }
    out.first          = entries.front().first;
    out.last           = entries.back().second;
    out.total          = cursor;
    const auto& tensor = *v3::tensor_object(directory.object(out.object));
    out.contiguous     = out.contiguous && cursor == elements(tensor.shape);
    return out;
}

void check_qwen38_nvfp4(const Reader& reader, const v3::Directory& directory) {
    expect(directory.objects.size() == 1246, "expected 1246 objects");
    expect(directory.bindings.size() == 1513, "expected 1513 bindings");
    expect(directory.uses.size() == 844, "expected 844 uses");
    expect(hex(reader.artifact_id()) == "19c9ec11085642f1bbe340cd7cf6c207",
           "artifact_id differs from the inspected file");

    std::size_t resources = 0;
    std::size_t tensors   = 0;
    for (const auto& object : directory.objects) {
        if (v3::object_is_tensor(object)) {
            ++tensors;
        } else {
            ++resources;
        }
    }
    expect(tensors == 1240 && resources == 6, "expected 1240 tensors and 6 resources");

    const auto& text = directory.component("text");
    expect(text.config.at("architectures").at(0) == "Qwen3_5ForCausalLM",
           "unexpected architecture");
    expect(text.config.at("num_hidden_layers") == 64, "unexpected layer count");
    expect(text.proposal.has_value() && text.proposal->indexed && text.proposal->rows == 131072,
           "expected the indexed 131072-row proposal domain");
    expect(text.resources.size() == 4, "expected four text frontend resources");
    expect(directory.component("vision").resources.size() == 2,
           "expected two vision frontend resources");

    // Fused parents consumed by the existing native ops must remain one object covering the whole
    // fused shape, in query/key/gate/value order.
    const auto attention = group_coverage(
        directory, {"text/layers/3/attention/query", "text/layers/3/attention/key",
                    "text/layers/3/attention/gate", "text/layers/3/attention/value"});
    expect(attention.contiguous && attention.total == 14336ULL * 5120ULL,
           "attention fusion does not cover one 14336x5120 parent");
    expect(v3::tensor_object(directory.object(attention.object))->format == "fp8_e4m3fn_row_bf16",
           "attention fusion parent is not row-scaled fp8");

    const auto gate_up =
        group_coverage(directory, {"text/layers/0/mlp/gate", "text/layers/0/mlp/up"});
    expect(gate_up.contiguous && gate_up.total == 34816ULL * 5120ULL,
           "mlp fusion does not cover one 34816x5120 parent");

    const auto gdn = group_coverage(directory, {"text/layers/0/gdn/query", "text/layers/0/gdn/key",
                                                "text/layers/0/gdn/value", "text/layers/0/gdn/z"});
    expect(gdn.contiguous && gdn.total == 16384ULL * 5120ULL,
           "gdn fusion does not cover one 16384x5120 parent");

    const auto ab = group_coverage(
        directory, {"text/layers/0/gdn/a_projection", "text/layers/0/gdn/b_projection"});
    expect(ab.contiguous && ab.total == 96ULL * 5120ULL,
           "gdn control fusion does not cover one 96x5120 parent");

    const auto vision_qkv = group_coverage(directory, {"vision/layers/0/attention/query",
                                                       "vision/layers/0/attention/key",
                                                       "vision/layers/0/attention/value"});
    expect(vision_qkv.contiguous && vision_qkv.total == 3456ULL * 1152ULL,
           "vision fusion does not cover one 3456x1152 parent");

    // The activation divisor is reachable only through the use record.
    const auto* use = directory.use("text/layers/0/mlp/gate", "text/layers/0/ffn_input");
    expect(use != nullptr, "mlp gate use is missing");
    expect(use->activation_policy == v3::ActivationPolicy::AllowA4,
           "mlp gate activation policy is not AllowA4");
    expect(use->auxiliaries.size() == 1 &&
               use->auxiliaries.begin()->first == "activation_input_divisor",
           "mlp gate has no activation input divisor auxiliary");
    std::size_t uses_with_auxiliary = 0;
    for (const auto& item : directory.uses) {
        if (!item.auxiliaries.empty()) { ++uses_with_auxiliary; }
    }
    expect(uses_with_auxiliary == 168, "expected 168 uses carrying an activation divisor");
}

void check_binder(const Reader& reader) {
    using ninfer::artifact::ArtifactError;
    using ninfer::artifact::Binder;
    const std::span<const std::uint64_t> scalar;

    Binder binder(reader);

    // A fused parent is bound from the logical parameters the native op consumes.
    const std::array<std::string_view, 4> attention_names{
        "text/layers/3/attention/query", "text/layers/3/attention/key",
        "text/layers/3/attention/gate", "text/layers/3/attention/value"};
    const std::array<std::uint64_t, 2> attention_shape{14336, 5120};
    const auto attention =
        binder.require_group(attention_names, NumericFormat::FP8_E4M3FN_ROW_BF16S,
                             StorageLayout::RowScaleV1, attention_shape);
    binder.materialize_on_device(attention);

    // One slice alone must not be handed to an op as if it were the whole parent.
    bool slice_rejected = false;
    const std::array<std::uint64_t, 2> key_shape{1024, 5120};
    try {
        (void)binder.require_tensor("text/layers/3/attention/key",
                                    NumericFormat::FP8_E4M3FN_ROW_BF16S, StorageLayout::RowScaleV1,
                                    key_shape);
    } catch (const ArtifactError& error) {
        slice_rejected = std::string(error.what()).find("require_group") != std::string::npos;
    }
    expect(slice_rejected, "binding a slice of a fused parent must be rejected");

    // Binding the same parent twice is a target bug, not an artifact bug.
    bool duplicate_rejected = false;
    try {
        (void)binder.require_group(attention_names, NumericFormat::FP8_E4M3FN_ROW_BF16S,
                                   StorageLayout::RowScaleV1, attention_shape);
    } catch (const ArtifactError&) { duplicate_rejected = true; }
    expect(duplicate_rejected, "binding one parent twice must be rejected");

    // A logical parameter that owns its whole object binds directly.
    const std::array<std::uint64_t, 1> hidden{5120};
    const auto final_norm = binder.require_tensor("text/final_norm", NumericFormat::BF16,
                                                  StorageLayout::ContiguousLeV1, hidden);
    binder.materialize_on_device(final_norm);

    // The activation divisor exists only as a use auxiliary.
    const auto divisor = binder.require_auxiliary(
        "text/layers/0/mlp/gate", "text/layers/0/ffn_input", "activation_input_divisor",
        NumericFormat::FP32, StorageLayout::ContiguousLeV1, scalar);
    binder.validate_only(divisor);
    bool policy_required = false;
    try {
        (void)binder.require_auxiliary("text/layers/0/mlp/gate", "text/layers/0/ffn_input",
                                       "weight_divisor", NumericFormat::FP32,
                                       StorageLayout::ContiguousLeV1, scalar);
    } catch (const ArtifactError& error) {
        policy_required = std::string(error.what()).find("no auxiliary") != std::string::npos;
    }
    expect(policy_required, "an unknown auxiliary role must be rejected");

    // Frontend resources are reached through the component's role map.
    const auto tokenizer = binder.require_component_resource(
        "text", "tokenizer.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
    expect(binder.payload(tokenizer).data.size() == 12809320, "unexpected tokenizer length");

    // Completeness is enforced: the selected target has not consumed every stored object yet.
    bool incomplete_rejected = false;
    try {
        (void)binder.finish();
    } catch (const ArtifactError& error) {
        incomplete_rejected =
            std::string(error.what()).find("was not consumed") != std::string::npos;
    }
    expect(incomplete_rejected, "finish() must reject a partially bound artifact");
}

} // namespace

int main(int argc, char** argv) {
    std::string path;
    if (argc > 1) {
        path = argv[1];
    } else if (const char* from_env = std::getenv("NINFER_V3_ARTIFACT")) {
        path = from_env;
    }
    if (path.empty()) {
        std::cout << "skipped: pass a .ninfer path or set NINFER_V3_ARTIFACT\n";
        return kSkipCode;
    }

    try {
        const Reader reader(path);
        expect(reader.version() == 3, "expected a v3 artifact");
        const auto& directory = reader.directory();

        std::cout << "artifact_id:   " << hex(reader.artifact_id()) << '\n'
                  << "name:          " << directory.name << '\n'
                  << "file_bytes:    " << reader.file_bytes() << '\n'
                  << "payload_start: " << reader.payload_offset() << '\n'
                  << "payload_bytes: " << directory.payload_bytes << '\n'
                  << "files:         " << directory.files.size() << '\n'
                  << "objects:       " << directory.objects.size() << '\n'
                  << "bindings:      " << directory.bindings.size() << '\n'
                  << "uses:          " << directory.uses.size() << '\n'
                  << "components:    ";
        for (const auto& [id, component] : directory.components) {
            std::cout << id << (component.target ? ("->" + *component.target + " ") : " ");
        }
        std::cout << '\n';

        std::map<std::string, std::size_t> formats;
        std::map<std::string, std::size_t> layouts;
        for (const auto& object : directory.objects) {
            if (const auto* tensor = v3::tensor_object(object)) {
                ++formats[tensor->format];
                ++layouts[tensor->layout];
            }
        }
        std::cout << "formats:       ";
        for (const auto& [name, count] : formats) { std::cout << name << '=' << count << ' '; }
        std::cout << "\nlayouts:       ";
        for (const auto& [name, count] : layouts) { std::cout << name << '=' << count << ' '; }
        std::cout << "\naliased parents: " << count_aliased_parents(directory) << '\n';

        // Every binding must resolve to a parsed object through the reader's own name index.
        for (const auto& [name, binding] : directory.bindings) {
            const std::string& id =
                binding.whole_object
                    ? v3::object_id(directory.object(binding.object))
                    : v3::object_id(directory.object(binding.parts.front().object));
            expect(reader.find(id) != nullptr, "binding '" + name + "' parent is not readable");
        }
        // Object descriptors stay in physical order with unique names.
        std::uint64_t cursor = 0;
        for (const auto& object : reader.objects()) {
            expect(ninfer::artifact::object_offset(object) >= cursor, "objects are out of order");
            cursor =
                ninfer::artifact::object_offset(object) + ninfer::artifact::object_bytes(object);
        }

        if (directory.name == "qwen3.8-27b") {
            check_qwen38_nvfp4(reader, directory);
            check_binder(reader);
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fail: " << error.what() << '\n';
        return 1;
    }
}
