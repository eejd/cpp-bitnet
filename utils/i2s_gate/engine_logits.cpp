// engine_logits: run a GGUF through libllama on the CPU (-ngl 0) and dump what the gate needs.
//
//   engine_logits --model m.gguf --threads N --cmds cmds.txt --out dir
//
// cmds.txt, one command per line:
//   L <idx> <id> <id> ...          logits of ALL positions of the given token sequence
//                                  -> <out>/<idx>.f32   (n_tokens x n_vocab, float32 LE)
//   G <idx> <n_new> <id> <id> ...  prefill the ids, then greedy-decode n_new tokens (argmax)
//                                  -> <out>/<idx>.ids   (generated ids, space separated)
//   K <idx> <add_special 0|1> <path>   tokenize the text file with the model's own tokenizer
//                                  -> <out>/<idx>.ids
// Always writes <out>/meta.json (n_vocab, system info = which SIMD paths libllama was built with,
// threads, model path). No sampling, no chat template, no KV reuse across commands.

#include "llama.h"

#include <execinfo.h>
#include <signal.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static void on_crash(int sig) {          // a crash inside libllama is a gate result, keep the stack
    void * bt[40];
    int n = backtrace(bt, 40);
    dprintf(2, "engine_logits: fatal signal %d, backtrace:\n", sig);
    backtrace_symbols_fd(bt, n, 2);
    _exit(128 + sig);
}

int main(int argc, char ** argv) {
    signal(SIGSEGV, on_crash); signal(SIGBUS, on_crash); signal(SIGABRT, on_crash);
    std::string model_path, cmds_path, out_dir = ".";
    int threads = 4;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = atoi(argv[++i]);
        else if (a == "--cmds" && i + 1 < argc) cmds_path = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--verbose") verbose = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || cmds_path.empty()) { fprintf(stderr, "usage: engine_logits --model m.gguf --cmds f --out dir [--threads N]\n"); return 2; }
    if (!verbose) llama_log_set(quiet_log, nullptr);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;                      // CPU lane only (Metal has no I2_S kernel)
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = 2048;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 3; }

    {
        std::string meta = std::string("{\"model\":\"") + model_path + "\",\"n_vocab\":" + std::to_string(n_vocab) +
                           ",\"threads\":" + std::to_string(threads) + ",\"system_info\":\"";
        for (const char * p = llama_print_system_info(); *p; p++) { if (*p == '"' || *p == '\\') meta += '\\'; meta += (*p == '\n' ? ' ' : *p); }
        meta += "\"}\n";
        std::ofstream(out_dir + "/meta.json") << meta;
    }

    auto decode = [&](const std::vector<llama_token> & ids, int pos0, bool all_logits) -> bool {
        llama_batch b = llama_batch_init((int) ids.size(), 0, 1);
        b.n_tokens = (int) ids.size();
        for (size_t i = 0; i < ids.size(); i++) {
            b.token[i] = ids[i]; b.pos[i] = pos0 + (int) i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
            b.logits[i] = all_logits || i + 1 == ids.size();
        }
        int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        return rc == 0;
    };

    std::ifstream cf(cmds_path);
    std::string line;
    int rc_all = 0;
    while (std::getline(cf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        std::string op; int idx;
        is >> op >> idx;
        llama_memory_clear(llama_get_memory(ctx), true);
        if (op == "L") {
            std::vector<llama_token> ids; long v; while (is >> v) ids.push_back((llama_token) v);
            if (!decode(ids, 0, true)) { fprintf(stderr, "decode failed for cmd %d\n", idx); rc_all = 4; continue; }
            std::ofstream o(out_dir + "/" + std::to_string(idx) + ".f32", std::ios::binary);
            for (size_t i = 0; i < ids.size(); i++) o.write((const char *) llama_get_logits_ith(ctx, (int) i), sizeof(float) * n_vocab);
        } else if (op == "G") {
            int n_new; is >> n_new;
            std::vector<llama_token> ids; long v; while (is >> v) ids.push_back((llama_token) v);
            std::vector<llama_token> gen;
            int pos = 0;
            if (!decode(ids, pos, false)) { fprintf(stderr, "decode failed for cmd %d\n", idx); rc_all = 4; continue; }
            pos = (int) ids.size();
            for (int s = 0; s < n_new; s++) {
                const float * lg = llama_get_logits_ith(ctx, -1);
                int best = 0; for (int t = 1; t < n_vocab; t++) if (lg[t] > lg[best]) best = t;
                gen.push_back(best);
                if (s + 1 < n_new && !decode({best}, pos++, false)) { rc_all = 4; break; }
            }
            std::ofstream o(out_dir + "/" + std::to_string(idx) + ".ids");
            for (size_t i = 0; i < gen.size(); i++) o << (i ? " " : "") << gen[i];
            o << "\n";
        } else if (op == "K") {
            int add_special; std::string path; is >> add_special >> path;
            std::ifstream tf(path); std::stringstream ss; ss << tf.rdbuf(); std::string text = ss.str();
            std::vector<llama_token> toks(text.size() + 16);
            int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), add_special != 0, false);
            if (n < 0) { fprintf(stderr, "tokenize failed\n"); rc_all = 4; continue; }
            std::ofstream o(out_dir + "/" + std::to_string(idx) + ".ids");
            for (int i = 0; i < n; i++) o << (i ? " " : "") << toks[i];
            o << "\n";
        } else { fprintf(stderr, "unknown cmd %s\n", op.c_str()); rc_all = 2; }
    }
    llama_free(ctx);
    llama_model_free(model);
    return rc_all;
}
