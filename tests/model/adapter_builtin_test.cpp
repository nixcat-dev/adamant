// tests/model/adapter_builtin_test.cpp - the program's own registration: the real adapter, once, and its load contract.
// Linking the models needs the CUDA runtime, so this runs where that is installed; no GPU work.
#include "basalt/model/adapter.hpp"
#include "basalt/models/builtin.hpp"

#include <cstdio>
#include <memory>

int main() {
    using namespace basalt;
    int failures = 0;
    const auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
        if (!ok) ++failures;
    };
    models::register_builtin_models();
    models::register_builtin_models();
    check(model::find_adapter("qwen4exp") != nullptr, "the qwen4exp adapter is registered");
    check(model::adapter_names().size() == 1, "registering twice keeps one adapter per name");
    if (const model::ModelAdapter* qw = model::find_adapter("qwen4exp")) {
        std::unique_ptr<model::LoadedModel> loaded;
        check(qw->load({}, loaded).status == model::Status::NotImplemented, "qwen4exp has no shared-server load: it runs its own start");
    }
    return failures ? 1 : 0;
}
