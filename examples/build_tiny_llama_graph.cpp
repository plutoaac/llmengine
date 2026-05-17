#include <iostream>

#include "minillm/minillm.h"

using namespace minillm;

int main() {
    // 1. Create Graph
    Graph graph;

    // 2. Create GraphBuilder
    GraphBuilder gb(graph);

    // 3. Build tiny decoder block
    TransformerConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = -1;       // dynamic
    cfg.vocab_size = 32000;
    cfg.hidden_size = 768;
    cfg.intermediate_size = 2048;
    cfg.num_heads = 12;
    cfg.num_kv_heads = 12;
    cfg.head_dim = 64;
    cfg.rms_norm_eps = 1e-6;

    TransformerGraphBuilder tb(gb);
    auto result = tb.build_tiny_decoder_block(cfg);
    if (!result) {
        std::cerr << "Failed to build graph: " << result.error().to_string() << "\n";
        return 1;
    }

    // 4. Validate
    auto st = graph.validate();
    if (!st.ok()) {
        std::cerr << "Graph validation failed: " << st.to_string() << "\n";
        return 1;
    }
    std::cout << "Graph validated OK\n\n";

    // 5. Dump graph
    std::cout << graph.dump() << "\n";

    // 6. MockExecutor compile & run
    auto backend = std::make_shared<CpuBackend>();
    MockExecutor exec(backend);

    st = exec.compile(graph);
    if (!st.ok()) {
        std::cerr << "Compile failed: " << st.to_string() << "\n";
        return 1;
    }
    std::cout << "Compile OK\n\n";

    RuntimeContext ctx;
    st = exec.run(ctx);
    if (!st.ok()) {
        std::cerr << "Run failed: " << st.to_string() << "\n";
        return 1;
    }
    std::cout << "Run OK\n";

    return 0;
}
