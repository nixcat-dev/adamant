// tests/model/adapter_registry_test.cpp - the adapter registry: lookup by architecture, the replaced guard and
// geometry read from the model's own metadata keys. Host only: no model and no runtime are linked.
#include "../core/gguf_fixture.hpp"
#include "basalt/model/adapter.hpp"
#include "meta_adapters.hpp"

#include <cstdio>
#include <filesystem>

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}
}  // namespace

int main() {
    using namespace basalt;
    test::register_meta_adapters();
    check(model::find_adapter("qwen4exp") != nullptr, "the qwen4exp adapter is registered");
    check(model::find_adapter("llama") == nullptr, "an unknown architecture has no adapter");

    const auto dir = std::filesystem::temp_directory_path() / "basalt_adapter_registry_test";
    std::filesystem::create_directories(dir);
    using namespace fixture;

    write(dir / "other.gguf", {str("general.architecture", "llama")}, {});
    check(model::check_architecture(GgufFile((dir / "other.gguf").string())).find("no model adapter") != std::string::npos,
          "an unregistered architecture is refused by name");

    write(dir / "qwen.gguf", {str("general.architecture", "qwen4exp")}, {});
    check(model::check_architecture(GgufFile((dir / "qwen.gguf").string())).find("block_count") != std::string::npos,
          "qwen4exp without its keys is refused");

    const core::ModelGeometry def;
    const std::vector<Kv> q = {str("general.architecture", "qwen4exp"),
                               u32("qwen4exp.block_count", (uint32_t) def.n_layers),
                               u32("qwen4exp.embedding_length", (uint32_t) def.n_embd),
                               u32("qwen4exp.expert_count", (uint32_t) def.n_expert),
                               u32("qwen4exp.expert_used_count", 12),
                               u32("qwen4exp.attention.head_count", (uint32_t) def.n_head),
                               u32("qwen4exp.attention.head_count_kv", (uint32_t) def.n_head_kv)};
    write(dir / "full.gguf", q, {});
    const GgufFile g((dir / "full.gguf").string());
    check(model::check_architecture(g).empty(), "qwen4exp with its keys validates");
    std::string err;
    const model::ModelAdapter* a = model::adapter_for(g, err);
    check(a != nullptr && std::string(a->arch()) == "qwen4exp", "adapter_for picks qwen4exp");
    if (a) {
        const model::ModelDesc d = a->describe(g);
        check(d.n_layers == def.n_layers && d.n_embd == def.n_embd && d.n_expert == def.n_expert && d.n_expert_used == 12 &&
                  d.n_ff_exp == def.n_ff,
              "describe() reads the geometry from the metadata");
    }
    std::filesystem::remove_all(dir);
    return failures ? 1 : 0;
}
