#pragma once

#include "artifact/binder.h"

#include <ninfer/targets/qwen3_6/frontend.h>

#include <string>

namespace ninfer::artifact {
class MaterializedArtifact;
}

namespace ninfer::targets::qwen3_6 {

struct FrontendResourcePlan {
    artifact::ObjectHandle tokenizer_json;
    artifact::ObjectHandle tokenizer_config_json;
    artifact::ObjectHandle chat_template_jinja;
    artifact::ObjectHandle generation_config_json;
    artifact::ObjectHandle preprocessor_config_json;
    artifact::ObjectHandle video_preprocessor_config_json;
    // A v2 artifact embeds the maintained template inside tokenizer_config.json; a v3 artifact
    // stores the upstream file verbatim and carries the maintained template separately.
    std::uint32_t artifact_version = 2;
};

struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
    std::uint32_t artifact_version = 2;
};

[[nodiscard]] FrontendResourcePlan bind_frontend_resources(artifact::Binder& binder);
// Version-3 form: resources are selected through each component's role map.
[[nodiscard]] FrontendResourcePlan bind_frontend_resources_v3(artifact::Binder& binder);
[[nodiscard]] FrontendResources take_frontend_resources(artifact::MaterializedArtifact& artifact,
                                                        const FrontendResourcePlan& plan);

} // namespace ninfer::targets::qwen3_6
